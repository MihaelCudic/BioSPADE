import yaml
from yaml.loader import SafeLoader

import torch

class Options():
    def __init__(self, file, yaml_file=None):
        self.initialized = False
        
        # Open the file and load the file
        with open(file) as f:
            self.opt = yaml.load(f, Loader=SafeLoader)   
        with open('options/base_options.yaml') as f:    
            self.opt.update(yaml.load(f, Loader=SafeLoader))
            
        # Open yaml file containing specific dataset details
        if yaml_file is None:
            yaml_file = self.opt['yaml_file']
        data_file = 'options/datasets/'+yaml_file+'_options.yaml'
        with open(data_file) as f:
            self.opt.update(yaml.load(f, Loader=SafeLoader))
            
        # Open yaml file contianing experiment changes
        changes_file = 'options/changes/'+yaml_file+'_changes.yaml'
        with open(changes_file, "r") as stream:
            self.changes = list(yaml.safe_load_all(stream))
        self.number_of_experiments = len(self.changes)
        if self.number_of_experiments == 0:
            self.run_all = False
    
    def save_options(self):
        out_file = 'checks/'+self.name+'/options.yaml'
        vars_list = vars(self)
        del vars_list['opt']
        del vars_list['changes']
        with open(out_file, 'w') as f:
            yaml.dump(vars_list, f)
            
    def set_experiment(self, exp):
        if not self.run_all:
            return
        
        if exp is not None:
            for key in self.changes[exp].keys():
                if not hasattr(self, key) and self.isTrain:
                    raise ValueError("%s not recognized as an option" % key)
                setattr(self, key, self.changes[exp][key])
        
    def initialize(self, exp=None):
        for key in self.opt.keys():
            setattr(self, key, self.opt[key])
            
        assert len(self.name)>0, f"\'name\' variable in options must be set to a string value"
        assert len(self.dataset_mode)>0, f"\'dataset_mode\' variable in options must be set to a string value"
        assert len(self.yaml_file)>0, f"\'yaml_file\' variable in options  must be set to a string value"
        if hasattr(self, 'mesh_root'):
            assert len(self.mesh_root)>0, f"\'mesh_root\' variable in options  must be set to a string value"
        if hasattr(self, 'stack_root'):
            assert len(self.stack_root)>0, f"\'stack_root\' variable in options must be set to a string value"
        
            
        if not self.run_all:
            self.number_of_experiments = 1
            self.number_of_instances = 1

        to_list = ['powers', 'frames']
        for key in to_list:
            if key in self.opt.keys():
                str_list = self.opt[key].split(',')
                int_list = [float(i) for i in str_list]
                setattr(self, key, int_list)
            
        # the default values for beta1 and beta2 differ by TTUR option
        no_TTUR = False if 'no_TTUR' not in self.opt else self.opt['no_TTUR']
        if no_TTUR:
            self.beta1=0.5
            self.beta2=0.999
            
        if not self.isTrain:
            if hasattr(self, 'test_batch_size'):
                self.batch_size = self.test_batch_size
            if hasattr(self, 'test_samples_per_instance'):
                self.samples_per_instance = self.test_samples_per_instance
            
        # set gpu ids
        str_ids = self.gpu_ids.split(',')
        self.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.gpu_ids.append(id)
        if len(self.gpu_ids) > 0:
            torch.cuda.set_device(self.gpu_ids[0])

        assert len(self.gpu_ids) == 0 or self.batch_size % len(self.gpu_ids) == 0, \
            "Batch size %d is wrong. It must be a multiple of # GPUs %d." \
            % (self.batch_size, len(self.gpu_ids))
        
        self.initialized = True