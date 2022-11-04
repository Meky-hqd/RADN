import os

def define_Dataset(stage,dataset_name):
    data_path = os.path.join('.', 'dataset', dataset_name, stage)
    if stage == 'train':
        if dataset_name in ['rain100L']:
            from dataset.DerainDataset import Dataset as D
            from dataset.DerainDataset import prepare_data_RainTrainL
            # prepare_data_RainTrainL(data_path=data_path,patch_size=96,stride=96)
            dataset = D(data_path)
            print('Training Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_name))
            return dataset
        elif dataset_name in ['rain100H']:
            from dataset.DerainDataset import Dataset as D
            from dataset.DerainDataset import prepare_data_RainTrainH
            data_path = os.path.join(data_path,'small')
            # prepare_data_RainTrainH(data_path=data_path, patch_size=96, stride=96)
            dataset = D(data_path)
            print('Training Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_name))
            return dataset
        elif dataset_name in ['rain800']:
            from dataset.DerainDataset import Dataset as D
            from dataset.DerainDataset import prepare_data_Rain800
            prepare_data_Rain800(data_path=data_path, patch_size=96, stride=96)
            dataset = D(data_path)
            print('Training Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_name))
            return dataset
        # 待添加
        else:
            raise NotImplementedError('Dataset [{:s}] is not found.'.format(dataset_name))


    elif stage == 'val':
        if dataset_name in ['rain100L']:
            #data_path = os.path.join('.', 'dataset', dataset_name, 'train')
            from dataset.DerainDataset import TestDataset as D
            dataset = D(data_path)
            print('Val Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_name))
            return dataset
        elif dataset_name in ['rain100H']:
            data_path = os.path.join(data_path,'small')
            ## rain12 用 rain100H来测试
            data_path = os.path.join('.', 'dataset', 'rain12')
            from dataset.DerainDataset import TestDataset as D
            dataset = D(data_path)
            print('Val Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_name))
            return dataset
        elif dataset_name in ['rain800']:
            from dataset.DerainDataset import TestDataset800 as D
            dataset = D(data_path)
            print('Val Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_name))
            return dataset
        # 待添加
        else:
            raise NotImplementedError('Dataset [{:s}] is not found.'.format(dataset_name))

    elif stage == 'pre_rain':
        if dataset_name in ['rain100H','rain100L','rain800']:
            #data_path = os.path.join('.', 'dataset', dataset_name, 'train')
            from dataset.DerainDataset import TestDataset as D
            dataset = D(data_path)
            print('Val Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_name))
            return dataset
        # 待添加
        else:
            raise NotImplementedError('Dataset [{:s}] is not found.'.format(dataset_name))

    else:
        raise ValueError(f'Stage value {stage} is invaild!')