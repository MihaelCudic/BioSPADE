## **Unpaired Mesh-to-Image Translation for 3D Fluorescent Microscopy Images of Neurons**

Code used to implement and evaluate BioSPADE

**Dependencies:**

    1. torch>=1.0.0
    2. torchvision
    3. dominate>=2.3.1
    4. dill
    5. scikit-image
    6. trimesh

**Data Download:**

- Real Fluorescence Microscopy Image Data

    1. 3DFM_AC and 3DFM_BCPop Datasets can be found 

- Meshes

    1. 2DSyn_BC meshes can be downloaded from eyewire.com [<a href="http://museum.eyewire.org/?neurons=60020,60033,60360,60374,60380,60383,60386,60388,60389,60404,60410,60414,60415,60421,60439,60442,60450,60458,60460,60462,60478,60488,60491,60497,60498,60504,60505,60510,60514,60519,60522,60523,60528,60541,60542,60615,60617,60618,60619,60620,60621&browse=1" target="_blank">link</a>]

    2. 3DSyn_GC meshes can be downloaded from eyewire.com [<a href="http://museum.eyewire.org/?neurons=17084,17097,17114,17140,20005,20011,20019,20071,20129,20140,20178,20181,20208,26023,26027,26028,26057,26068,26089,26125,26141,26148,26191,30002,30003&browse=1" target="_blank">link</a>]
    
    3. 3DFM_AC meshes can be downloaded from eyewire.com <a href="http://museum.eyewire.org/?neurons=20196,20204,70164,70171,70176,70182,70183,70185,70189,70193,70203,70205,70222,70225,70225,70229,70230,70237&browse=1" target="_blank">link</a>]

**Instructions:**

    1. Download the data
    
    2. Set 'mesh_root' and 'stack_root' in options/datasets/[dataset_file].yaml to the respective path of your data
    
    3. Set 'name' in options/base_options.yaml to your experiment name
    
    4. Set 'dataset_mode' and 'fmBCPop3D' to either ['synBC2D'|'synGC3D'|'fmAC3D'|'fmBCPop3D']
    
    5. Run code from train.ipynb
    
If you want want to evaluate algorithms:

    6. Run code from train-segmentation.ipynb
    
    7. Run code from test-[dataset].ipynb
    
    
    
