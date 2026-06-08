import os

import app.model as m

# TO DO
# if it is not learning then try setting real comparison to 0.9 instead of 1

# is buffer optimal, cuases 4sec additional time per epoch
# if buffer fails then update every 2-5 batches discriminator

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(project_root, 'data')

def intra_gender():
    saved_model_sf1_sf2 = os.path.join(project_root, 'saved_models', 'VCC2SF1_VCC2SF2_epoch_400.pth')

    model = m.Model('VCC2SF1', 'VCC2SF2')
    # model.train()
    # model.evaluate(saved_model_sf1_sf2)
    model.test(saved_model_sf1_sf2)

    # eval_file_sf1 = os.path.join(data_dir, 'evaluation_data', 'transformed_audio', 'VCC2SF1', '30004.npz')
    # model.voiceToTarget('VCC2SF1', 'VCC2SF2', eval_file_sf1, saved_model_sf1_sf2)
    
    # eval_file_sf2 = os.path.join(data_dir, 'evaluation_data', 'transformed_audio', 'VCC2SF2', '30004.npz')
    # model.voiceToTarget('VCC2SF2', 'VCC2SF1', eval_file_sf2, saved_model_sf1_sf2)

def inter_gender():
    saved_model_sf1_sm4 = os.path.join(project_root, 'saved_models', 'VCC2SF1_VCC2SM4_epoch_200.pth')

    model = m.Model('VCC2SF1', 'VCC2SM4')
    model.train()
    model.evaluate(saved_model_sf1_sm4)
    model.test(saved_model_sf1_sm4)

    eval_file_sf1 = os.path.join(data_dir, 'evaluation_data', 'transformed_audio', 'VCC2SF1', '30004.npz')
    model.voiceToTarget('VCC2SF1', 'VCC2SM4', eval_file_sf1, saved_model_sf1_sm4)
    
    eval_file_sm4 = os.path.join(data_dir, 'evaluation_data', 'transformed_audio', 'VCC2SM4', '30004.npz')
    model.voiceToTarget('VCC2SM4', 'VCC2SF1', eval_file_sm4, saved_model_sf1_sm4)

def main():
    intra_gender()    

if __name__ == '__main__':
    main()