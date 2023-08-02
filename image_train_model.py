import os
import argparse
import numpy as np
import tensorflow as tf
from image_maml import MetaCNN


def train(model, dataset, sess, saver):
    for epoch in range(iterations):
        if "meta" in model_type:
            idx = np.random.choice(800, update_batch_size, replace=False)
            batch_x, batch_y = dataset[0][idx], dataset[1][idx]
            inputa, labela = batch_x, batch_y
            batch_x, batch_y = dataset[2][idx], dataset[3][idx]
            inputb, labelb = batch_x, batch_y
            feed_dict = {model.inputa: inputa, model.inputb: inputb,
                         model.labela: labela, model.labelb: labelb}
        else:
            raise Exception(NotImplementedError)

        if epoch % 100 == 0:
            model_file = save_dir + "/" + model_type + "/model_" + str(epoch)
            saver.save(sess, model_file)
            if "meta" in model_type:
                res = sess.run([model.total_rmse1, model.total_rmse2], feed_dict)
            print(epoch, res)
        else:
            if "meta" in model_type:
                sess.run([model.metatrain_op], feed_dict)
            elif "pretrain" in model_type:
                sess.run([model.pretrain_op], feed_dict)


def main():
    tf.set_random_seed(1234)

    # get data
    img_size=(255, 255)
    x_a_train = np.random.rand(800, img_size[0], img_size[1], 8)
    y_a_train = np.random.rand(800, img_size[0], img_size[1], 1)
    x_b_train = np.random.rand(800, img_size[0], img_size[1], 8)
    y_b_train = np.random.rand(800, img_size[0], img_size[1], 1)
    dataset = [x_a_train, y_a_train, x_b_train, y_b_train]

    # get model
    print(model_type, "meta" in model_type)
    if "meta" in model_type:
        model = MetaCNN(img_size=img_size, dim_input=dim_input,
                        dim_output=dim_output, filter_num=64,
                         update_lr=update_lr, meta_lr=meta_lr,
                         meta_batch_size=len(cities),
                         update_batch_size=update_batch_size,
                         test_num_updates=test_num_updates)
    else:
        raise Exception(NotImplementedError)

    model.construct_model()

    sess = tf.InteractiveSession()
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=10)

    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()

    print("Training:", model_type)
    train(model, dataset, sess, saver)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cities', type=str, default='nyc,dc')
    parser.add_argument('--save_dir', type=str, default='./models')
    parser.add_argument('--model_type', type=str, default='metacnn')

    parser.add_argument('--update_batch_size', type=int, default=128)
    parser.add_argument('--test_num_updates', type=int, default=1)
    parser.add_argument('--threshold', type=float, default=0)
    parser.add_argument('--meta_lr', type=float, default=1e-5)
    parser.add_argument('--update_lr', type=float, default=1e-5)
    # parser.add_argument('--cluster_loss_weight', type=float)
    # parser.add_argument('--mem_dim', type=int, default=8)

    parser.add_argument('--iterations', type=int, default=20000)
    parser.add_argument('--gpu_id', type=str, default="3")

    dim_output = 1
    dim_input = 8

    args = parser.parse_args()

    cities = args.cities.split(',')
    save_dir = args.save_dir
    model_type = args.model_type

    update_batch_size = args.update_batch_size
    test_num_updates = args.test_num_updates
    threshold = args.threshold

    cluster_loss_weight = args.cluster_loss_weight
    mem_dim = args.mem_dim

    meta_lr = args.meta_lr
    update_lr = args.update_lr

    iterations = args.iterations

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    main()
