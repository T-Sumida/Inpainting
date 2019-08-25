# coding:utf-8
import tensorflow as tf
import inpaint_module

WIDTH = 256
HEIGHT = 256
BATCH_SIZE = 32
EPOCHS = 1000000
mask_size = [128, 64, 96]


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
    loss_d_red = tf.reduce_mean(
        -tf.reduce_sum(tf.log(d_real_red) +
                       tf.log(tf.ones(BATCH_SIZE, tf.float32) - d_fake_red),
                       axis=1))
    loss_D = loss_d_red

    loss_gen = -tf.reduce_mean(d_fake_red)
    loss_Gan = loss_gen

    Loss_s_re = tf.reduce_mean(tf.abs(decoder_i - Y))
    Loss_hat = tf.reduce_mean(tf.abs(decoder_c - Y))

    A = tf.image.rgb_to_yuv((input_red_image+1)/2.0)
    A_Y = tf.to_int32(A[:, :, :, 0:1]*255.0)

    B = tf.image.rgb_to_yuv((Y+1)/2.0)
    B_Y = tf.to_int32(B[:, :, :, 0:1]*255.0)

    ssim = tf.reduce_mean(tf.image.ssim(A_Y, B_Y, 255.0))
    alpha = IT/EPOCHS
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

    for i in range(EPOCHS):


if __name__ == "__main__":
    main()
