# --init--

width=224
height=224
channel=3

my_epochs=10
my_batch_size=64

num_classes = 4
algorithms_name=['DenseNet201', 'VGG16', 'ResNet101']

alg_num=len(algorithms_name)
accuracy_array=np.zeros(alg_num)
precision_array=np.zeros(alg_num)
recall_array=np.zeros(alg_num)
f1_score_array=np.zeros(alg_num)
