source activate shaperecon
conda config --add channels conda-forge
conda install shapely rtree pyembree numpy scipy==1.3.0
conda install -c conda-forge scikit-image
pip install trimesh[all]
