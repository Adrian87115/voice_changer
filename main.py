import model as m

def main():
    model = m.Model("VCC2SF1", "VCC2SF2")
    # model.train(load_state = False)
    # model.evaluate()
    # model.voiceToTarget("VCC2SF1", "VCC2SF2", "evaluation_data/transformed_audio/VCC2SF1/30004.npz")
    model.voiceToTarget("VCC2SF2", "VCC2SF1", "evaluation_data/transformed_audio/VCC2SF2/30004.npz")

if __name__ == "__main__":
    main()
