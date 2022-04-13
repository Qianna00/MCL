from setuptools import setup, find_packages


setup(name='MCL',
      version='1.0.0',
      description='Multi-level contrastive learning for Unsupervised Vessel Re-Identification',
      author='Qian Zhang',
      author_email='zhq9669@gmail.com',
      install_requires=[
          'numpy', 'torch', 'torchvision',
          'six', 'h5py', 'Pillow', 'scipy',
          'scikit-learn', 'metric-learn', 'faiss_gpu'],
      packages=find_packages(),
      keywords=[
          'Unsupervised Learning',
          'Contrastive Learning',
          'Vessel Re-identification'
      ])
