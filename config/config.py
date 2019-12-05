class Config(object):
	env = 'default'
	backbone = 'resnet18'
	classify = 'softmax'
	num_classes = 9223
	metric = 'arc_margin'
	easy_margin = False
	use_se = False
	loss = 'focal_loss'
	bottleneck_size = 128
	
	display = False
	finetune = False
	
	path_to_model = 'checkpoints'
	save_interval = 10

	train_batch_size = 8  # batch size
	test_batch_size = 60

	optimizer = 'sgd'

	use_gpu = True  # use GPU or not
	gpu_id = '0, 1'
	num_workers = 4  # how many workers for loading data
	print_freq = 100  # print info every N batch

	debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
	result_file = 'result.csv'

	max_epoch = 50
	lr = 1e-1  # initial learning rate
	lr_step = 10
	lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
	weight_decay = 5e-4
