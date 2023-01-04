import pdb
import tensorflow as tf
import numpy as np
import scipy.misc 

class Logger(object):
    
    def __init__(self, log_dir ):
        """Create a summary writer logging to log_dir."""
        
        self.train_writer = tf.summary.create_file_writer(log_dir + "/train")
        self.test_writer = tf.summary.create_file_writer(log_dir + "/eval")
        print(f"Logs are written to {log_dir}")

        self.loss = tf.Variable(0.0)
        tf.summary.scalar("loss", self.loss)

        # self.merged = tf.summary.merge_all()

        # self.session = tf.compat.v1.InteractiveSession()
        # self.session.run(tf.compat.v1.global_variables_initializer())


    def scalar_summary(self, train_loss, test_loss, epoch, train_logs,test_logs):
        """Log a scalar variable."""

        # summary = self.session.run(self.merged, {self.loss: train_loss})
        # self.train_writer.add_summary(summary, step) 
        # pdb.set_trace()
        print('saving logs', epoch,train_logs, test_logs)
        with self.train_writer.as_default():
            tf.summary.scalar(name='train loss', step=epoch, data=train_loss)

            for key, value in train_logs.items():
                tf.summary.scalar(name=key, step=epoch, data=value)
            
            self.train_writer.flush()

        # summary = self.session.run(self.merged, {self.loss: test_loss})
        # self.test_writer.add_summary(summary, step) 
        # self.test_writer.flush()
        with self.test_writer.as_default():
            tf.summary.scalar(name='test loss', step=epoch, data=test_loss)

            for key, value in test_logs.items():
                tf.summary.scalar(name=key, step=epoch, data=value)

            self.test_writer.flush()

        
        
