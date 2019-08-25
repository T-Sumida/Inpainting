# coding:utf-8
import tensorflow as tf
import inpaint_module
import cv2
from data_generator import DataGenerator

WIDTH = 256
HEIGHT = 256
BATCH_SIZE = 4
EPOCHS = 1000000
BATCH_MAX = 2
SAVE_COUNTER = 2
SAVE_DIR = "./tmp/"


def main():
    # 変数準備
    X = tf.placeholder(tf.float32, [BATCH_SIZE, HEIGHT, WIDTH, 3])
    Y = tf.placeholder(tf.float32, [BATCH_SIZE, HEIGHT, WIDTH, 3])
    mask = tf.placeholder(tf.float32, [BATCH_SIZE, HEIGHT, WIDTH, 3])
    IT = tf.placeholder(tf.float32)

    # モデルの用意
    en_input = tf.concat([X, mask], 3)
    x = inpaint_module.encoder(en_input, 'encoder')
    x_con = inpaint_module.contextual_attention(
        x, x, mask, 3, 50, 'contextual_attention')
    decoder_i = inpaint_module.decoder(
        x_con, 'decoder', WIDTH, HEIGHT, reuse=False)
    decoder_c = inpaint_module.decoder(
        x, 'decoder', WIDTH, HEIGHT, reuse=True)
    input_red_image = decoder_i * (1-mask) + Y*mask
    d_real_red = inpaint_module.discriminator(Y, 'red', reuse=False)
    d_fake_red = inpaint_module.discriminator(
        input_red_image, 'red', reuse=True)

    # ロスの定義
    loss_d_red = tf.reduce_mean(d_fake_red) + tf.reduce_mean(1-d_real_red)
    loss_D = loss_d_red

    loss_gen = -tf.reduce_mean(d_fake_red)
    loss_Gan = loss_gen

    Loss_s_re = tf.reduce_mean(tf.abs(decoder_i - Y))
    Loss_hat = tf.reduce_mean(tf.abs(decoder_c - Y))
    alpha = IT/(EPOCHS*BATCH_MAX)
    loss_G = 0.1*loss_Gan + 10*Loss_s_re + 5*(1-alpha) * Loss_hat

    # 最適化
    var_D = [v for v in tf.global_variables() if v.name.startswith('red')]
    var_G = [v for v in tf.global_variables() if v.name.startswith(
        'encoder') or v.name.startswith('decoder') or v.name.startswith('contextual_attention')]
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimize_D = tf.train.AdamOptimizer(
            learning_rate=0.0004, beta1=0.5, beta2=0.9).minimize(loss_D,
                                                                 var_list=var_D)
        optimize_G = tf.train.AdamOptimizer(
            learning_rate=0.0001, beta1=0.5, beta2=0.9).minimize(loss_G,
                                                                 var_list=var_G)

    # 実行
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    config.gpu_options.allow_growth = False
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()

    gen = DataGenerator(WIDTH, HEIGHT, BATCH_SIZE)
    valid_image, valid_masks, valid_image_m = gen.get_data()

    with open(SAVE_DIR+"log.csv", 'w') as f:
        f.write("epoch,d_loss,g_loss,recon_loss")
    for e in range(EPOCHS):
        e_loss1, e_loss2, e_loss3 = 0, 0, 0
        for b in range(BATCH_MAX):
            image, masks, image_m = gen.get_data()
            if image.shape[0] != BATCH_SIZE:
                continue
            _, loss1 = sess.run([optimize_D, loss_D], feed_dict={
                                X: image_m, Y: image, mask: masks})
            _, loss2, loss3 = sess.run([optimize_G, loss_G, Loss_s_re],
                                       feed_dict={
                                       X: image_m, Y: image, mask: masks, IT: b+1})
            e_loss1 += loss1 / BATCH_MAX
            e_loss2 += loss2 / BATCH_MAX
            e_loss3 += loss3 / BATCH_MAX
        print('Epoch : %d\tD Loss = %.5f\tG Loss = %.5f\tRecon Loss = %.5f' % (
            e, e_loss1, e_loss2, e_loss3))
        log_msg = "\n{},{},{},{}".format(e, e_loss1, e_loss2, e_loss3)
        with open(SAVE_DIR+"log.csv", 'a') as f:
            f.write(log_msg)
        if e % SAVE_COUNTER == 0:
            s_m_path = SAVE_DIR + "model".format(e)
            saver.save(sess, s_m_path)
            img_sample = sess.run([input_red_image], feed_dict={
                X: valid_image_m, Y: valid_image, mask: valid_masks})
            for k in range(3):
                save_img = img_sample[0][k][:][:][:]
                s_path = SAVE_DIR + "{:05}_{}.png".format(e, k)
                cv2.imwrite(s_path, save_img)


if __name__ == "__main__":
    main()
