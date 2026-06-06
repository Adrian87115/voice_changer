import os

import app.model as m

# TO DO
# ADD METRICS FOR EVALUATION AS IN THE RP

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'data')
    # saved_model_sf1_sf2 = os.path.join(project_root, 'saved_models', 'VCC2SF1_VCC2SF2_epoch_200.pth')

    model = m.Model('VCC2SF1', 'VCC2SF2')
    model.train()
    # model.evaluate()
    
    # eval_file_sf1 = os.path.join(data_dir, 'evaluation_data', 'transformed_audio', 'VCC2SF1', '30004.npz')
    # model.voiceToTarget('VCC2SF1', 'VCC2SF2', eval_file_sf1, saved_model_sf1_sf2)
    
    # eval_file_sf2 = os.path.join(data_dir, 'evaluation_data', 'transformed_audio', 'VCC2SF2', '30004.npz')
    # model.voiceToTarget('VCC2SF2', 'VCC2SF1', eval_file_sf2, saved_model_sf1_sf2)


    # model = m.Model('VCC2SF1', 'VCC2SM4')
    # model.train(load_state = True)
    # model.evaluate()
    
    # saved_model_sf1_sm4 = os.path.join(project_root, 'saved_models', 'VCC2SF1_VCC2SM4_epoch_100.pth')

    # eval_file_sf1_sm4 = os.path.join(data_dir, 'evaluation_data', 'transformed_audio', 'VCC2SF1', '30004.npz')
    # model.voiceToTarget('VCC2SF1', 'VCC2SM4', eval_file_sf1_sm4, saved_model_sf1_sm4)
    
    # eval_file_sm2_sf4 = os.path.join(data_dir, 'evaluation_data', 'transformed_audio', 'VCC2SM4', '30004.npz')
    # model.voiceToTarget('VCC2SM4', 'VCC2SF1', eval_file_sm4_sf1, saved_model_sf1_sm4)

if __name__ == '__main__':
    main()