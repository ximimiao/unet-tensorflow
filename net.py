from models.unet import *
import logging
import tensorflow as tf
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


class Model(object):
    def __init__(self, dataprovider):
        self.dataprovider = dataprovider
        self.size = 256
        self.x = tf.placeholder(tf.float32, [None, self.size, self.size, 3])
        self.y = tf.placeholder(tf.int32, [None, self.size, self.size, 1])
        self.logits = unet(x=self.x,batch_norm=True,n_class=5,features=32)
        self.predict = tf.argmax(self.logits, axis=3)
        self.loss = self.get_loss()
        self.total_acc, self.plant_acc, self.load_acc, \
        self.build_acc, self.water_acc = self.get_acc()
        # self.total_acc = self.get_acc()

    def get_loss(self):
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.squeeze(self.y),
                logits=self.logits
            )
        )
        return loss

    def get_acc(self):
        y_flat = tf.reshape(self.y,[-1])
        pre_flat = tf.reshape(self.predict,[-1])
        mat = tf.confusion_matrix(
            labels=y_flat,
            predictions=pre_flat,
            num_classes=5
        )
        total_acc = tf.reduce_sum(tf.diag_part(mat)) / tf.reduce_sum(mat)
        plant_acc = mat[1,1]/tf.reduce_sum(mat[1])
        load_acc =  mat[2,2]/tf.reduce_sum(mat[2])
        build_acc = mat[3,3]/tf.reduce_sum(mat[3])
        water_acc = mat[4,4]/tf.reduce_sum(mat[4])

        return total_acc, plant_acc, load_acc, build_acc, water_acc
        # return total_acc

    def save(self, sess, model_path):

        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path)
        logging.info('Done!')
        return save_path



class Train(object):
    def __init__(self, dataprovider, model, batch_size=32):
        self.dataprovider = dataprovider
        self.batch_size = batch_size
        self.model = model


    def initialize(self, sess):
        self.learning_rate = tf.Variable(0.001)
        tf.summary.scalar('loss',self.model.loss)
        tf.summary.scalar('total_acc',self.model.total_acc)
        tf.summary.scalar('plant_acc',self.model.plant_acc)
        tf.summary.scalar('load_acc',self.model.load_acc)
        tf.summary.scalar('build_acc',self.model.build_acc)
        tf.summary.scalar('water_acc',self.model.water_acc)
        self.summary_op = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter('log/train', sess.graph)
        self.test_writer = tf.summary.FileWriter('log/test')
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.model.loss)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())


    def train(self, iters=100000):
        logging.info("Start optimization")
        with tf.Session() as sess:
            self.initialize(sess)
            for step in range(iters):
                x_tr, y_tr = self.dataprovider.next_batch_tr(self.batch_size)
                x_test, y_test = self.dataprovider.next_batch_test(self.batch_size)
                if step % 100 == 0:
                    tr_list = sess.run([self.model.loss,
                                        self.summary_op,
                                        self.model.total_acc,
                                        self.optimizer],
                                       feed_dict={self.model.x: x_tr, self.model.y: y_tr})
                    test_list = sess.run([self.model.loss,
                                          self.summary_op,
                                          self.model.total_acc],
                                         feed_dict={self.model.x: x_test, self.model.y: y_test})
                    self.train_writer.add_summary(tr_list[1],step)
                    self.test_writer.add_summary(test_list[1],step)
                    logging.info("iter {:}, tr_loss: {:.4f}, tr_acc {:.4f}, test_loss: {:.4f}, test_acc {:.4f}"
                                 .format(step, tr_list[0], tr_list[2], test_list[0], test_list[2]))

                else:

                    _, summary_tr = sess.run([self.optimizer,self.summary_op],feed_dict={
                        self.model.x:x_tr,
                        self.model.y:y_tr
                    })
                    self.train_writer.add_summary(summary_tr,step)
                    summary_test = sess.run(self.summary_op,feed_dict={
                        self.model.x:x_test,
                        self.model.y:y_test
                    })
                    self.test_writer.add_summary(summary_test,step)


