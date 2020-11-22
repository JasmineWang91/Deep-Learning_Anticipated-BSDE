import numpy as np
import tensorflow as tf
import scipy
from sklearn.preprocessing import StandardScaler
import time
import numpy.matlib
import matplotlib.pyplot as plt
import scipy.stats as st

d = 1
N = 100
T = 1
X0 = np.full(d,0)
I = np.full(d,1)
I_mat = np.diag(np.full(d,1))
H = T/N
sqrth=np.sqrt(H)
sqrtd=np.sqrt(d)

n_epochs = 10
Mbatch = 1000
n_batches = 100
M_training = n_epochs*Mbatch*n_batches # Size of the training set
MBatchValidation = 1000 # Size of the Validation set
M_validation = n_epochs*MBatchValidation

###############################     Parameter Setting    ################################################
norm= lambda x:np.dot(x,I)


def g_tf(x):  #Terminal value (tf)
    return tf.reduce_sum(x, axis=1, keepdims=True)

def sum_tf(x):
    return tf.reduce_sum(x, keepdims=True)

np.random.seed(1)
NoiseTraining=np.random.normal(0,1,(M_training,d))
NoiseValidation=np.random.normal(0,1,(M_validation,d))

X = np.zeros((N+1,d))
L = np.zeros((N+1,1))
V = np.zeros((N+1,1))
noise = np.random.normal(0, 1, (N, d))
X[0] = 2 * np.dot(2*I_mat, (1/d)*I_mat)
L[0] = max(0, np.dot(2*I_mat, (1/d)*I_mat)-1)
V[0] = 1.5*max(0, np.dot(2*I_mat, (1/d)*I_mat)-1)

for n in range(N):
    X[n+1] = X[n] + (1.1*H + 0.1*np.dot(noise[n],I_mat)) * X[n]
    #L[n+1] = max(0, X[n+1]-1)
    #V[n+1] = 1.5*max(0, X[n+1]-1)

def f_tf(x, y, z):
    return 3*g_tf(x)+tf.abs(2*y+3*g_tf(z))

def F_tf(x, y, z, B):
    return y - f_tf(x, y, z)*H + sqrth*g_tf(z*B)

###########################################################################################################
##################################     NN    ##############################################################
###########################################################################################################
n_inputs = d
n_hidden1_Y = d+10
n_hidden2_Y = d+10
n_outputs_Y = 1
n_hidden1_Z = d+10
n_hidden2_Z = d+10
n_outputs_Z = d
scale = 0.001
nbOuterLearning=20
min_decrease_rate = 0.05

Y_op = np.zeros((M_training, 1))
Yn_op11 = np.zeros((N+1, 1))
Yn_plot0=np.zeros((N,1))
Yn_plot1=np.zeros((N,1))
Yn_plot2=np.zeros((N,1))
Zn_plot0=np.zeros((N,d))
Zn_plot1=np.zeros((N,d))
Zn_plot2=np.zeros((N,d))
Y_opN = 1.2* max(0, np.dot(2*I_mat, (1/d)*I_mat)-1)
Yn_op11[N] = Y_opN
print("Estimation of U/Y at step %d: " % (N), Y_opN)

###########################################################################################################
def TrainYnnZnnN(n):  # Train the optimal control and value function at time N-1. We do not use the pre-training trick.
    assert (n == N - 1)
    tf.reset_default_graph()
    Ynext_op = tf.placeholder(tf.float64, shape=(None,1), name="Ynext_op"+str(n+1))
    Ln = tf.placeholder(tf.float64, shape=(None, 1), name="Lower" + str(n))
    Vn = tf.placeholder(tf.float64, shape=(None, 1), name="Upper" + str(n))
    Xn = tf.placeholder(tf.float64, shape=(None, n_inputs), name="X" + str(n))
    Bn = tf.placeholder(tf.float64, shape=(None, n_inputs), name="Noise" + str(n))
    zeroY = tf.placeholder(tf.float64, shape=(None, 1), name="zero" + str(n))
    Xsc = tf.placeholder(tf.float64, shape=(None,n_inputs), name="Xsc")
    learning_rate = tf.placeholder(tf.float64, name="learning_rate")
    regularizer = tf.contrib.layers.l2_regularizer(scale)
    he_init = tf.contrib.layers.variance_scaling_initializer()

    with tf.name_scope("dnn_Y"):
        hidden1_Y=tf.layers.dense(Xsc, n_hidden1_Y, name="hidden1_Y"+str(n), activation=tf.nn.tanh, kernel_initializer=he_init,kernel_regularizer=regularizer)
        hidden2_Y=tf.layers.dense(hidden1_Y, n_hidden2_Y, name="hidden2_Y"+str(n), activation=tf.nn.tanh, kernel_initializer=he_init,kernel_regularizer=regularizer)
        output_Y=tf.layers.dense(hidden2_Y, n_outputs_Y, name="output_Y"+str(n), kernel_initializer=he_init,kernel_regularizer=regularizer)

    with tf.name_scope("dnn_Z"):
        hidden1_Z=tf.layers.dense(Xsc,n_hidden1_Z, name="hidden1_Z"+str(n), activation=tf.nn.tanh, kernel_initializer=he_init,kernel_regularizer=regularizer)
        hidden2_Z=tf.layers.dense(hidden1_Z, n_hidden2_Z, name="hidden2_Z"+str(n), activation=tf.nn.tanh, kernel_initializer=he_init,kernel_regularizer=regularizer)
        output_Z=tf.layers.dense(hidden2_Z, n_outputs_Z, name="output_Z"+str(n), kernel_initializer=he_init,kernel_regularizer=regularizer)

    with tf.name_scope("loss_YZ"):
        reglosses_YZ=tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term_YZ=tf.contrib.layers.apply_regularization(regularizer, reglosses_YZ)
        lossYZ=tf.reduce_mean(tf.square(Ynext_op-F_tf(Xn, output_Y, output_Z, Bn)))

    with tf.name_scope("train_Y"):
        train_vars_Y =tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="hidden1_Y"+str(n)+"|hidden2_Y"+str(n)+"|output_Y"+str(n))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        training_op_Y= optimizer.minimize(lossYZ+reg_term_YZ, var_list=train_vars_Y)

    with tf.name_scope("train_Z"):
        train_vars_Z = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="hidden1_Z" + str(n) + "|hidden2_Z" + str(n) + "|output_Z" + str(n))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        training_op_Z= optimizer.minimize(lossYZ+reg_term_YZ, var_list=train_vars_Z)

    #Yn_op = output_Y + tf.maximum(zeroY, Ln - output_Y) - tf.maximum(zeroY, output_Y - Vn)
    Yn_op = output_Y

    init = tf.global_variables_initializer()
    saver=tf.train.Saver()
    with tf.Session() as sess:
        init.run()
        init_learning_rate=0.001
        loss_hist=[]

        for epoch in range(n_epochs):
            onesarray1 = np.ones((MBatchValidation, 1))
            ep1 = epoch * MBatchValidation
            ep2 = (epoch + 1) * MBatchValidation
            val_loss = lossYZ.eval(feed_dict={Xsc: NoiseValidation[ep1: ep2], Ynext_op: Y_opN*onesarray1, Xn: X[n]*onesarray1, Bn: noise[n]*onesarray1})
            loss_hist.append(val_loss)
            #print("Loss of %d th epoch: %f" %(epoch, val_loss))

            for batch in range(n_batches):
                ind1=n_batches*epoch*Mbatch + batch*Mbatch
                ind2=n_batches*epoch*Mbatch + (batch+1)*Mbatch
                onesarray2 = np.ones((Mbatch, 1))
                sess.run(training_op_Y, feed_dict={learning_rate: init_learning_rate, Xsc: NoiseTraining[ind1:ind2], Ynext_op: Y_opN*onesarray2, Xn: X[n]*onesarray2, Bn: noise[n]*onesarray2})
                sess.run(training_op_Z, feed_dict={learning_rate: init_learning_rate, Xsc: NoiseTraining[ind1:ind2], Ynext_op: Y_opN*onesarray2, Xn: X[n]*onesarray2, Bn: noise[n]*onesarray2})

                if epoch % nbOuterLearning == 0:
                    mean_loss = np.mean(loss_hist)
                    if epoch > 0:
                        # print("mean_loss=", mean_loss)
                        # print("last_loss_check", last_loss_check)
                        decrease_rate = (last_loss_check - mean_loss) / last_loss_check
                        # print("decrease_rate=", decrease_rate)
                        if decrease_rate < min_decrease_rate:
                            init_learning_rate = np.maximum(1e-6, init_learning_rate / 2)
                            # print("learningRate decreased to ", init_learning_rate)
                    last_loss_check = mean_loss
                    loss_hist = []

        Yn_output = output_Y.eval(feed_dict={Xsc: NoiseTraining})
        Zn_output = output_Z.eval(feed_dict={Xsc: NoiseTraining})
        Yn_optimal = Yn_op.eval(feed_dict={output_Y: Yn_output, Ln: L[n] * np.ones((M_training, 1)), Vn: V[n] * np.ones((M_training, 1)), zeroY: np.zeros((M_training, 1))})
        Yn_op11[n]=Yn_optimal[0]
        for i in range(M_training):
            Y_op[i] = Yn_optimal[i]
        print("Estimation of U at step %d: " % (n), Yn_optimal[0:5])
        print("Estimation of Z at step %d: " % (n), Zn_output[0:5])
        save_path = saver.save(sess, "saver/Vfinal" + str(n) + ".ckpt")

TrainYnnZnnN(N-1)

###########################################################################################################
def TrainYnnZnn(n):  # Train the optimal control and value function at time N-1. We do not use the pre-training trick.
    tf.reset_default_graph()
    Y_inuptnext = tf.placeholder(tf.float64, shape=(None, 1), name="Y_next" + str(n + 1))
    Ynext_op = tf.placeholder(tf.float64, shape=(None,1), name="Ynext_op"+str(n+1))
    Ln = tf.placeholder(tf.float64, shape=(None, 1), name="Lower" + str(n))
    Vn = tf.placeholder(tf.float64, shape=(None, 1), name="Upper" + str(n))
    Xn = tf.placeholder(tf.float64, shape=(None, n_inputs), name="X" + str(n))
    Bn = tf.placeholder(tf.float64, shape=(None, n_inputs), name="Noise" + str(n))
    zeroY = tf.placeholder(tf.float64, shape=(None, 1), name="zero" + str(n))
    Xsc = tf.placeholder(tf.float64, shape=(None,n_inputs), name="Xsc")
    learning_rate = tf.placeholder(tf.float64, name="learning_rate")
    regularizer = tf.contrib.layers.l2_regularizer(scale)
    he_init = tf.contrib.layers.variance_scaling_initializer()

    with tf.name_scope("dnn_Ynext"):
        hidden1_Y=tf.layers.dense(Y_inuptnext, n_hidden1_Y, name="hidden1_Y"+str(n+1), activation=tf.nn.tanh)
        hidden2_Y=tf.layers.dense(hidden1_Y, n_hidden2_Y, name="hidden2_Y"+str(n+1), activation=tf.nn.tanh)
        output_Ynext=tf.layers.dense(hidden2_Y, n_outputs_Y, name="output_Y"+str(n+1))

    with tf.name_scope("dnn_Y"):
        hidden1_Y=tf.layers.dense(Xsc, n_hidden1_Y, name="hidden1_Y"+str(n), activation=tf.nn.tanh, kernel_initializer=he_init,kernel_regularizer=regularizer)
        hidden2_Y=tf.layers.dense(hidden1_Y, n_hidden2_Y, name="hidden2_Y"+str(n), activation=tf.nn.tanh, kernel_initializer=he_init,kernel_regularizer=regularizer)
        output_Y=tf.layers.dense(hidden2_Y, n_outputs_Y, name="output_Y"+str(n), kernel_initializer=he_init,kernel_regularizer=regularizer)

    with tf.name_scope("dnn_Z"):
        hidden1_Z=tf.layers.dense(Xsc,n_hidden1_Z, name="hidden1_Z"+str(n), activation=tf.nn.tanh, kernel_initializer=he_init,kernel_regularizer=regularizer)
        hidden2_Z=tf.layers.dense(hidden1_Z, n_hidden2_Z, name="hidden2_Z"+str(n), activation=tf.nn.tanh, kernel_initializer=he_init,kernel_regularizer=regularizer)
        output_Z=tf.layers.dense(hidden2_Z, n_outputs_Z, name="output_Z"+str(n), kernel_initializer=he_init,kernel_regularizer=regularizer)

    with tf.name_scope("loss_YZ"):
        reglosses_YZ=tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_term_YZ=tf.contrib.layers.apply_regularization(regularizer, reglosses_YZ)
        lossYZ=tf.reduce_mean(tf.square(output_Ynext-F_tf(Xn, output_Y, output_Z, Bn)))

    with tf.name_scope("train_Y"):
        train_vars_Y =tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="hidden1_Y"+str(n)+"|hidden2_Y"+str(n)+"|output_Y"+str(n))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        training_op_Y= optimizer.minimize(lossYZ+reg_term_YZ, var_list=train_vars_Y)

    with tf.name_scope("train_Z"):
        train_vars_Z = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="hidden1_Z" + str(n) + "|hidden2_Z" + str(n) + "|output_Z" + str(n))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        training_op_Z= optimizer.minimize(lossYZ+reg_term_YZ, var_list=train_vars_Z)

    #Yn_op = output_Y + tf.maximum(zeroY, Ln - output_Y) - tf.maximum(zeroY, output_Y - Vn)
    Yn_op = output_Y

    init = tf.global_variables_initializer()
    saver=tf.train.Saver()
    with tf.Session() as sess:
        init.run()
        init_learning_rate=0.001
        loss_hist=[]

        for epoch in range(n_epochs):
            onesarray1 = np.ones((MBatchValidation, 1))
            ep1 = epoch * MBatchValidation
            ep2 = (epoch + 1) * MBatchValidation

            val_loss = lossYZ.eval(feed_dict={Xsc: NoiseValidation[ep1: ep2], Y_inuptnext: Y_op[ep1: ep2], Xn: X[n]*onesarray1, Bn: noise[n]*onesarray1})

            loss_hist.append(val_loss)
            #print("Loss of %d th epoch: %f" %(epoch, val_loss))

            for batch in range(n_batches):
                ind1=n_batches*epoch*Mbatch + batch*Mbatch
                ind2=n_batches*epoch*Mbatch + (batch+1)*Mbatch
                onesarray2 = np.ones((Mbatch, 1))
                sess.run(training_op_Y, feed_dict={learning_rate: init_learning_rate, Xsc: NoiseTraining[ind1:ind2], Y_inuptnext: Y_op[ind1:ind2], Xn: X[n] * onesarray2, Bn: noise[n] * onesarray2})
                sess.run(training_op_Z, feed_dict={learning_rate: init_learning_rate, Xsc: NoiseTraining[ind1:ind2], Y_inuptnext: Y_op[ind1:ind2], Xn: X[n] * onesarray2, Bn: noise[n] * onesarray2})

                if epoch % nbOuterLearning == 0:
                    mean_loss = np.mean(loss_hist)
                    if epoch > 0:
                        # print("mean_loss=", mean_loss)
                        # print("last_loss_check", last_loss_check)
                        decrease_rate = (last_loss_check - mean_loss) / last_loss_check
                        # print("decrease_rate=", decrease_rate)
                        if decrease_rate < min_decrease_rate:
                            init_learning_rate = np.maximum(1e-6, init_learning_rate / 2)
                            # print("learningRate decreased to ", init_learning_rate)
                    last_loss_check = mean_loss
                    loss_hist = []

        Yn_output = output_Y.eval(feed_dict={Xsc: NoiseTraining})
        Zn_output = output_Z.eval(feed_dict={Xsc: NoiseTraining})
        Yn_optimal = Yn_op.eval(feed_dict={output_Y: Yn_output, Ln: L[n] * np.ones((M_training, 1)), Vn: V[n] * np.ones((M_training, 1)), zeroY: np.zeros((M_training, 1))})

        Yn_plot0[n] = Yn_optimal[0]
        Yn_plot1[n] = Yn_optimal[1]
        Yn_plot2[n] = Yn_optimal[2]
        Zn_plot0[n] = Zn_output[0]
        Zn_plot1[n] = Zn_output[1]
        Zn_plot2[n] = Zn_output[2]

        Yn_op11[n]=Yn_optimal[0]

        #print(Yn_optimal[0:3])
        #print(Zn_output[0:3])

        for i in range(M_training):
            Y_op[i] = Yn_optimal[i]
        #print(st.describe(Optimal))

        if n==0:
            print("Estimation of U at step %d: " % (n), Yn_optimal[0:5])
            print("Estimation of Z at step %d: " % (n), Zn_output[0:5])
            print(st.describe(Yn_optimal))

            plt.figure()
            t_plot = np.arange(n*H, T, H)
            Y_plot0 = np.array(Yn_plot0[n::])
            Y_plot1 = np.array(Yn_plot1[n::])
            Y_plot2 = np.array(Yn_plot2[n::])
            plt.xlabel('x')
            plt.ylabel('u(x)')
            plt.plot(t_plot, Y_plot0, color='red')
            plt.plot(t_plot, Y_plot1, color='blue')
            plt.plot(t_plot, Y_plot2, color='green')
            plt.show()

            if d==1:
                plt.figure()
                t_plot = np.arange(n*H, T, H)
                Z_plot0 = np.array(Zn_plot0[n::])
                Z_plot1 = np.array(Zn_plot1[n::])
                Z_plot2 = np.array(Zn_plot2[n::])
                plt.xlabel('x')
                plt.ylabel('z(x)')
                plt.plot(t_plot, Z_plot0, color='red')
                plt.plot(t_plot, Z_plot1, color='blue')
                plt.plot(t_plot, Z_plot2, color='green')
                plt.show()

            plt.figure()
            t_plot = np.arange(n*H, T, H)
            L_plot = np.array(L[n:N])
            V_plot = np.array(V[n:N])
            Y_plot = np.array(Yn_plot0[n::])
            plt.xlabel('t')
            plt.ylabel('u(x)')
            plt.plot(t_plot, Y_plot, color='red', label='U')
            plt.plot(t_plot, L_plot, color='skyblue', label='Lower obstacle')
            plt.plot(t_plot, V_plot, color='yellow', label='Upper obstacle')
            plt.legend()
            plt.show()


###########################################################################################################
##################################     NN    ##############################################################
###########################################################################################################
start_time=time.time()
for n in range(N-1,-1,-1):
    TrainYnnZnn(n)
run_time=time.time()- start_time
print("Running time: ",run_time)
noise = np.random.normal(0, 1, (N, d))

