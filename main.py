import model as m
def main():
    model = m.Model()
    model.train()
    # avg_eval_loss, accuracy = model.evaluate()
    # print(avg_eval_loss, accuracy)
    model.voiceToTarget("VCC2TF2", "evaluation_data/resized_audio/VCC2SF1/30001.npz")



if __name__ == "__main__":
    main()

# the last hope is maybe to train in the order that is mentioned in the paper, so not in one loop everything at once