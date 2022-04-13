from __future__ import print_function, absolute_import
import time
import collections
import torch
import torch.nn.functional as F
from .utils.meters import AverageMeter


class MCLTrainer(object):

    def __init__(self, encoder, alpha=1.0, temperature=0.05):
        super(MCLTrainer, self).__init__()
        self.encoder = encoder
        self.alpha = alpha
        self.temperature = temperature

    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400):
        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()
        losses_cc = AverageMeter()
        losses_tmp = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            # process inputs
            imgs1, imgs2, labels, indexes = self._parse_data(inputs)
            batch_size = imgs1.size(0)
            imgs = torch.cat([imgs1, imgs2], dim=0)

            # forward
            f_out = self._forward(imgs)
            f_out1 = f_out[:batch_size, ::]
            z = F.normalize(f_out, dim=1)

            # z1, z2 = torch.split(z, [batch_size, batch_size], dim=0)
            # z = torch.cat((z1.unsqueeze(1), z2.unsqueeze(1)), dim=1)

            # print("f_out shape: {}".format(f_out.shape))
            # compute loss with the hybrid memory
            # loss = self.memory(f_out, indexes)
            batch_centers = collections.defaultdict(list)
            for instance_feature, index in zip(f_out1, labels.tolist()):
                batch_centers[index].append(instance_feature.unsqueeze(0))

            centroids = collections.defaultdict()
            for index, features in batch_centers.items():
                features_tensor = torch.cat(features, dim=0)
                centroid = F.normalize(features_tensor.mean(dim=0).unsqueeze(0), dim=1)
                centroids[index] = centroid
            # centroids_cat = torch.cat([centroid for index, centroid in centroids.items()], dim=0)
            loss = []
            for index, label in enumerate(labels.tolist()):
                positive_centroid = centroids[label]
                temp_centroids = centroids.copy()
                temp_centroids.pop(label)
                negative_centroids = torch.cat([centroid for index, centroid in temp_centroids.items()], dim=0)
                centroids_ = torch.cat([positive_centroid, negative_centroids], dim=0)
                outputs = z[index].unsqueeze(0).mm(centroids_.t())
                outputs /= 0.05
                targets = outputs.new_zeros((1,), dtype=torch.long)
                loss_ = F.cross_entropy(outputs, targets)
                loss.append(loss_)
                # centroids_.append(torch.cat([positive_centroid, negative_centroids], dim=0).unsqueeze(0))
            # centroids_ = torch.cat(centroids_, dim=0)
            # outputs = torch.matmul(f_out.unsqueeze(1), centroids_.permute(0, 2, 1)).squeeze()
            loss_cc = sum(loss)
            loss_tmp = self.tmp_loss(z, labels, batch_size)
            loss = self.alpha * loss_cc + loss_tmp

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())
            losses_cc.update(loss_cc.item())
            losses_tmp.update(loss_tmp.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss_cc {:.3f} ({:.3f})\t'
                      'Loss_tmp {:.3f} ({:.3f})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              losses_cc.val, losses_cc.avg,
                              losses_tmp.val, losses_tmp.avg))

    def _parse_data(self, inputs):
        imgs1, imgs2, _, pids, _, indexes = inputs
        return imgs1.cuda(), imgs2.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)

    def tmp_loss(self, features, label, batch_size):
        N = batch_size
        # features = torch.cat(torch.unbind(features, dim=1), dim=0)
        logit = torch.matmul(features, features.t())

        mask = 1 - torch.eye(2 * N, dtype=torch.uint8).cuda()
        logit = torch.masked_select(logit, mask == 1).reshape(2 * N, -1)

        # label = concat_all_gather(labels)
        label = label.view(-1, 1)
        label = label.repeat(2, 1)

        label_mask = label.eq(label.t()).float()
        # label_mask = label_mask.repeat(2, 2)
        is_neg = 1 - label_mask

        # 2N x (2N - 1)
        pos_mask = torch.masked_select(label_mask.bool(),
                                       mask == 1).reshape(2 * N, -1)
        neg_mask = torch.masked_select(is_neg.bool(),
                                       mask == 1).reshape(2 * N, -1)

        # rank, world_size = get_dist_info()
        # size = int(2 * N / world_size)

        # pos_mask = torch.split(pos_mask, [size] * world_size, dim=0)[rank]
        # neg_mask = torch.split(neg_mask, [size] * world_size, dim=0)[rank]
        # logit = torch.split(logit, [size] * world_size, dim=0)[rank]

        n = logit.size(0)
        loss = []

        for i in range(n):
            if label[i] == -1:
                continue
            pos_inds = torch.nonzero(pos_mask[i] == 1, as_tuple=False).view(-1)
            neg_inds = torch.nonzero(neg_mask[i] == 1, as_tuple=False).view(-1)

            loss_single_img = []
            for j in range(pos_inds.size(0)):
                positive = logit[i, pos_inds[j]].reshape(1, 1)
                negative = logit[i, neg_inds].unsqueeze(0)
                _logit = torch.cat((positive, negative), dim=1)
                _logit /= self.temperature
                _label = _logit.new_zeros((1,), dtype=torch.long)
                _loss = torch.nn.CrossEntropyLoss()(_logit, _label)
                loss_single_img.append(_loss)
            loss.append(sum(loss_single_img) / pos_inds.size(0))

        loss_ = sum(loss)
        # loss /= logit.size(0)
        loss_ /= len(loss)
        return loss_
