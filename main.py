import model as m
def main():
    model = m.Model()
    model.train()
    # avg_eval_loss, accuracy = model.evaluate()
    # print(avg_eval_loss, accuracy)
    # model.voiceToTarget("VCC2TF2", "evaluation_data/resized_audio/VCC2SF1/30001.npz")

    # archiecture with atribute c - target label
    # eval data
    # proper learning loop - check


if __name__ == "__main__":
    main()