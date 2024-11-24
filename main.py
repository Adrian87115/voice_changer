import model as m
def main():
    model = m.Model("VCC2SF1", "VCC2SM2")
    # model.train(load_state = True)
    # avg_eval_loss_xy, avg_eval_loss_yx, avg_cycle_loss, avg_adv_loss = model.evaluate()
    # print(f"xy loss: {avg_eval_loss_xy:.4f}, yx loss: {avg_eval_loss_yx:.4f}, Cycle loss: {avg_cycle_loss:.4f}, Adv loss: {avg_adv_loss:.4f}")
    # model.voiceToTarget("VCC2SM2", "VCC2SF1", "evaluation_data/transformed_audio/VCC2SM2/30004.npz")
    # model.voiceToTarget("VCC2SF1", "VCC2SM2", "evaluation_data/transformed_audio/VCC2SF1/30004.npz")

if __name__ == "__main__":
    main()
