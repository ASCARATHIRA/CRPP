import sys
import tensorflow as tf
import numpy as np
import math
from tensorflow.python import debug as tf_debug

processed = sys.argv[1]
rank=sys.argv[2]

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in xrange(0, len(l), n):
        yield l[i:i + n]

with open(processed, "r") as f:
	content = f.readlines()
	f.close()
with open(rank, "r") as f2:
	content2 = f2.readlines()
	f2.close()

intervals=[]
givenrankintervals = []
no_intervals = 100

for line in content2:
	tokens=line.split("\t")
	t1=float(tokens[0])/10000
	t2=float(tokens[1])/10000
	givenrankintervals.append((t1,t2))
	ranks = [int(x) for x in tokens[2:]]
	#print ranks
	
	
L=len(givenrankintervals)
p=len(content)/7

tlast = givenrankintervals[L-1][1]

interval_size = tlast/no_intervals
marker = 0
for i in range(no_intervals):
	intervals.append((marker, marker+interval_size))
	marker = marker + interval_size
intervalcounts=[[0 for x in range(no_intervals)] for x in range(p)]
pred_intervalcounts=[[0 for x in range(no_intervals)] for x in range(p)]


test_t=[]


h_series = [[] for x in range(p)]
integral = [[] for x in range(p)]

num_epochs = 100
truncated_backprop_length = 15
echo_step = 3
batch_size = 5


batchk_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
batcht_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
	
init_h=tf.placeholder(tf.float32, (batch_size))

omega = tf.constant([0.01], shape=(batch_size,))
a_list = []
b_list = []
c_list = []
alpha_list = []
beta_list = []
gamma_list = []
t_series = [[] for x in range(p)]
k_series = [[] for x in range(p)]
G_loss = [[] for x in range(p)]
a = [[] for x in range(p)]
b = [[] for x in range(p)]
c = [[] for x in range(p)]

l=0
for n in range(p):

	alpha=tf.Variable([0.001 for x in range(batch_size)])
	beta=tf.Variable([0.001 for x in range(batch_size)])
	gamma=tf.Variable([0.001 for x in range(batch_size)])	
	a[n]=tf.Variable([0.001 for x in range(batch_size)])
	b[n]=tf.Variable([0.001 for x in range(batch_size)])
	c[n]=tf.Variable([0.001 for x in range(batch_size)])
		
	
	str_k=content[l]
	k=[float(ki) for ki in str_k.split(",")]
	str_train=content[l+1].rstrip()
	train_t=[float(ti)/10000 for ti in str_train.split(",")] 
	str_test=content[l+6].rstrip()
	test_t=[float(ti)/10000 for ti in str_test.split(",")]
	l=l+7
	
	first_training_instance = train_t[0]
	last_training_instance = train_t[-1]
	first_training_interval = 0
	last_training_interval = no_intervals - 1
	
	while train_t[0]>intervals[first_training_interval][1]:
		first_training_interval = first_training_interval+1
	while train_t[-1]<intervals[last_training_interval][0]:
		last_training_interval = last_training_interval-1
		
	trainingintervals = intervals[:last_training_interval]
		
	counter = 0
	for i in range(len(train_t)):
		while train_t[i]>intervals[counter][1]:
			counter = counter+1
		intervalcounts[n][counter] = intervalcounts[n][counter] + 1
		
		
	for i in range(len(test_t)):
		while counter<no_intervals and test_t[i]>intervals[counter][1]:
			counter = counter+1
		if counter==no_intervals:
			break
		intervalcounts[n][counter] = intervalcounts[n][counter] + 1
	
	last_test_interval = no_intervals - 1				
	while intervalcounts[n][last_test_interval]==0:
		last_test_interval = last_test_interval-1
	#print intervalcounts[n][last_test_interval], intervals[last_test_interval]
	no_training_intervals = last_training_interval-first_training_interval
	no_test_intervals = no_intervals - no_training_intervals
			
	
	total_series_length = len(k)
	num_batches = total_series_length//batch_size//truncated_backprop_length
	
	
	k_series[n] = tf.unstack(batchk_placeholder, axis=1)
	t_series[n] = tf.unstack(batcht_placeholder, axis=1)
	
	
	# Forward pass
	current_h = init_h
	for i in range(len(k_series)):
		current_k = k_series[n][i]
		current_t = t_series[n][i]
		
		numerator = -omega*current_t
		denominator = current_k
		frac = tf.truediv(numerator,denominator)
		kernel = tf.exp(frac) 
		next_h = tf.exp((alpha*current_h + beta*kernel + gamma))
		h_series[n].append(next_h)
		current_h = next_h
	
	
	G_loss[n] = tf.reduce_sum(tf.exp(a[n]*h_series[n][-1]+b[n]*t_series[n][-1]+c[n])/b[n] - tf.exp(a[n]*h_series[n][-1]+b[n]*t_series[n][0]+c[n])/b[n]) - sum([tf.reduce_sum(a[n]*h_series[n][-1]+b[n]*(t_series[n][k1+1]-t_series[n][k1])+c[n]) for k1 in range(len(t_series)-1)])
		
	regularizer = tf.add_n([tf.nn.l2_loss(a[n]),tf.nn.l2_loss(b[n]),tf.nn.l2_loss(c[n]),tf.nn.l2_loss(alpha),tf.nn.l2_loss(beta),tf.nn.l2_loss(gamma)])
	
	total_loss = G_loss[n]+regularizer
	
	train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)
	
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		
		loss_list = []
		h_list = []
		last_h = np.ones((batch_size))
		last_batchk = np.ones((batch_size, truncated_backprop_length))
		last_batcht = np.ones((batch_size, truncated_backprop_length))
		#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
		#sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
		#writer = tf.summary.FileWriter("/home/supritam/Documents/aaai/code", sess.graph)

    	
		for epoch_idx in range(num_epochs):
			_current_h = np.ones((batch_size))
			        		        		
			#print("New data, epoch", epoch_idx)

			for batch_idx in range(num_batches):
				start_idx = batch_idx * batch_size * truncated_backprop_length
				end_idx = start_idx + batch_size * truncated_backprop_length
				
				batchk = list(chunks(k[0:3], truncated_backprop_length))		
				batchk = list(chunks(k[start_idx:end_idx], truncated_backprop_length))
				batcht = list(chunks(train_t[start_idx:end_idx], truncated_backprop_length))
				
				#---debugging statements---
				#print batchk, _current_h
								            			
				_a, _b, _c, _total_loss, _train_step, _current_h, = sess.run(
                	[a[n], b[n], c[n], total_loss, train_step, current_h],
                	feed_dict={
						batchk_placeholder:batchk,
						batcht_placeholder:batcht,
						init_h:_current_h,
						
				})
				
				last_h = _current_h
				last_batchk = batchk
				last_batcht = batcht

				loss_list.append(_total_loss)
				h_list.append(_current_h)
				
				#if batch_idx%100 == 0:
				#print("Step", batch_idx, "Loss", _total_loss)
				#print("a", _a)
				#print("b", _b)
				#print("c", _c)
		
		
		
		
		saved_a, saved_b, saved_c, saved_alpha, saved_beta, saved_gamma = sess.run([a[n], b[n], c[n], alpha, beta, gamma], feed_dict={
                    	batchk_placeholder:last_batchk,
                    	batcht_placeholder:last_batcht,
                    	init_h:last_h,
                    	
                })
		a_list.append(saved_a)
		b_list.append(saved_b)
		c_list.append(saved_c)
		alpha_list.append(saved_alpha)
		beta_list.append(saved_beta)
		gamma_list.append(saved_gamma)
		
		for i in range(first_training_interval, last_training_interval+1):
			t1, t2 = intervals[i]
			pred_intervalcounts[n][i] = math.exp(sum(saved_a*last_h + saved_b*t2 + saved_c))/sum(saved_b) - math.exp(sum(saved_a*last_h + saved_b*t1 + saved_c))/sum(saved_b)
		
#Convex optimization based competition model

l = tf.Variable([[0.001 for x in range(p)] for y in range(p)])
D_loss = []
predplaceholders = tf.placeholder(tf.float32, [p,p])

predmatrix = [[0 for x in range(p)] for x in range(p)]
for i in range(p-1):
	for j in range(i+1, p):
		for n in range(last_training_interval+1):
			D_loss.append(l[i][j]*(w[i][j]*predplaceholders[i][j]+xi[i][j]))
			if intervalcounts[i][n] > intervalcounts[j][n]:
				predmatrix[i][j] = (pred_intervalcounts[i][n]-pred_intervalcounts[j][n]-1)
			elif intervalcounts[j][n] > intervalcounts[i][n]:
				predmatrix[i][j] = (pred_intervalcounts[j][n]-pred_intervalcounts[i][n]-1)
				
total_D_loss = tf.reduce_sum(D_loss)
D_train_step = tf.train.AdagradOptimizer(0.3).minimize(total_D_loss)
	
ADV_EPOCHS = 100
for counter in range(ADV_EPOCHS):
	_l = []
	for epoch in range(1000):
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			_l, _D_train_step = sess.run([l, D_train_step], feed_dict={
                    	predplaceholders:predmatrix
                    	                 	
                })
    
	D_loss = []
	for i in range(p-1):
		for j in range(i+1, p):
			for n in range(last_training_interval+1):
				if intervalcounts[i][n] > intervalcounts[j][n]:
					D_loss.append(_l[i][j]*(tf.reduce_sum(tf.exp(a[i]*h_series[i][-1]+b[i]*(intervals[n][1]-intervals[n][0])+c[i])) - tf.exp(a[j]*h_series[j][-1]+b[j]*(intervals[n][1]-intervals[n][0])+c[j])))
					
				elif intervalcounts[j][n] > intervalcounts[i][n]:
					D_loss.append(_l[i][j]*(tf.reduce_sum(tf.exp(a[j]*h_series[j][-1]+b[j]*(intervals[n][1]-intervals[n][0])+c[j])) - tf.exp(a[i]*h_series[i][-1]+b[i]*(intervals[n][1]-intervals[n][0])+c[i])))
					
					
	
	
	l = 0
	for n in range(p):
		alpha=tf.Variable([0.001 for x in range(batch_size)])
		beta=tf.Variable([0.001 for x in range(batch_size)])
		gamma=tf.Variable([0.001 for x in range(batch_size)])	
		a[n]=tf.Variable([0.001 for x in range(batch_size)])
		b[n]=tf.Variable([0.001 for x in range(batch_size)])
		c[n]=tf.Variable([0.001 for x in range(batch_size)])
		
	
		str_k=content[l]
		k=[float(ki) for ki in str_k.split(",")]
		str_train=content[l+1].rstrip()
		train_t=[float(ti)/10000 for ti in str_train.split(",")] 
		str_test=content[l+6].rstrip()
		test_t=[float(ti)/10000 for ti in str_test.split(",")]
		l=l+7
	
		first_training_instance = train_t[0]
		last_training_instance = train_t[-1]
		first_training_interval = 0
		last_training_interval = no_intervals - 1
	
		while train_t[0]>intervals[first_training_interval][1]:
			first_training_interval = first_training_interval+1
		while train_t[-1]<intervals[last_training_interval][0]:
			last_training_interval = last_training_interval-1
		
		trainingintervals = intervals[:last_training_interval]
		
		counter = 0
		for i in range(len(train_t)):
			while train_t[i]>intervals[counter][1]:
				counter = counter+1
			intervalcounts[n][counter] = intervalcounts[n][counter] + 1
		
		
		for i in range(len(test_t)):
			while counter<no_intervals and test_t[i]>intervals[counter][1]:
				counter = counter+1
			if counter==no_intervals:
				break
			intervalcounts[n][counter] = intervalcounts[n][counter] + 1
	
		last_test_interval = no_intervals - 1				
		while intervalcounts[n][last_test_interval]==0:
			last_test_interval = last_test_interval-1
		#print intervalcounts[n][last_test_interval], intervals[last_test_interval]
		no_training_intervals = last_training_interval-first_training_interval
		no_test_intervals = no_intervals - no_training_intervals
			
	
		total_series_length = len(k)
		num_batches = total_series_length//batch_size//truncated_backprop_length
	
	
		
		regularizer = tf.add_n([tf.nn.l2_loss(a[n]),tf.nn.l2_loss(b[n]),tf.nn.l2_loss(c[n]),tf.nn.l2_loss(alpha),tf.nn.l2_loss(beta),tf.nn.l2_loss(gamma)])
	
		G_loss[n] = tf.reduce_sum(tf.exp(a[n]*h_series[n][-1]+b[n]*t_series[n][-1]+c[n])/b[n] - tf.exp(a[n]*h_series[n][-1]+b[n]*t_series[n][0]+c[n])/b[n]) - sum([tf.reduce_sum(a[n]*h_series[n][-1]+b[n]*(t_series[k+1]-t_series[k])+c[n]) for k in range(len(t_series)-1)])
	
		adv_loss = tf.reduce_sum(D_loss) + tf.reduce_sum(G_loss[n])
		
		total_loss = adv_loss+regularizer
	
		train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)
	
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
		
			loss_list = []
			h_list = []
			last_h = np.ones((batch_size))
			last_batchk = np.ones((batch_size, truncated_backprop_length))
			last_batcht = np.ones((batch_size, truncated_backprop_length))
			#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
			#sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
			#writer = tf.summary.FileWriter("/home/supritam/Documents/aaai/code", sess.graph)

    	
			for epoch_idx in range(num_epochs):
				_current_h = np.ones((batch_size))
			        		        		
				#print("New data, epoch", epoch_idx)

				for batch_idx in range(num_batches):
					start_idx = batch_idx * batch_size * truncated_backprop_length
					end_idx = start_idx + batch_size * truncated_backprop_length
            			
					batchk = list(chunks(k[start_idx:end_idx], truncated_backprop_length))
					batcht = list(chunks(train_t[start_idx:end_idx], truncated_backprop_length))
				
					#---debugging statements---
					#print batchk, _current_h
								            			
					_a, _b, _c, _total_loss, _train_step, _current_h, = sess.run(
						[a[n], b[n], c[n], total_loss, train_step, current_h],
						feed_dict={
							batchk_placeholder:batchk,
							batcht_placeholder:batcht,
							init_h:_current_h,
						
					})
				
					last_h = _current_h
					last_batchk = batchk
					last_batcht = batcht

					
					#if batch_idx%100 == 0:
					#print("Step", batch_idx, "Loss", _total_loss)
					#print("a", _a)
					#print("b", _b)
					#print("c", _c)
		
		
		
		
			saved_a, saved_b, saved_c, saved_alpha, saved_beta, saved_gamma = sess.run([a[n], b[n], c[n], alpha, beta, gamma], feed_dict={
                    	batchk_placeholder:last_batchk,
                    	batcht_placeholder:last_batcht,
                    	init_h:last_h,
                    	
                })
			
			for i in range(first_training_interval, last_training_interval+1):
				t1, t2 = intervals[i]
				pred_intervalcounts[n][i] = math.exp(saved_a*last_h + saved_b*t2 + saved_c)/saved_b - math.exp(saved_a*last_h + saved_b*t1 + saved_c)/saved_b
	
for n in range(p):
	cumulative = 0
	cumulative_pred = 0
	for i in range(first_training_interval, last_training_interval):
		t1, t2 = intervals[i]
		actual = intervalcounts[n][i]
		cumulative = cumulative+actual
		pred = pred_intervalcounts[n][i]
		cumulative_pred = cumulative_pred+pred
		trainAPE = trainAPE + abs(cumulative_pred-cumulative)/cumulative
		#print "Dataset:", dataset, "Interval:", t1, t2, "MAPE:", trainAPE/(i+1)
	
	trainMAPE = trainAPE/no_training_intervals
	#print "Training MAPE:", trainMAPE
	
	testAPE = trainAPE
			
	for i in range(last_training_interval, last_test_interval+1):
	#for i in range(last_training_interval, no_intervals):
		t1, t2 = intervals[i]
		actual = intervalcounts[n][i]
		cumulative = cumulative+actual
		pred = pred_intervalcounts[n][i]
		cumulative_pred = cumulative_pred+pred
		#print "Ntrue:", cumulative, "Npred:", cumulative_pred
		testAPE = testAPE + abs(cumulative_pred-cumulative)/cumulative
		#print "Dataset:", dataset, "Interval:", t1, t2, "MAPE:", testAPE/(i+1)
	
	testMAPE = testAPE/no_intervals
	#print "Test MAPE:", testMAPE
		
	MAPEsum = MAPEsum + testMAPE
	
	for x in range(rankintervals):
		rankcounts[n][x] = sum(intervalcounts[n][last_training_interval+x*no_test_intervals/rankintervals:last_training_interval+(x+1)*no_test_intervals/rankintervals])
		pred_rankcounts[n][x] = sum(pred_intervalcounts[n][last_training_interval+x*no_test_intervals/rankintervals:last_training_interval+(x+1)*no_test_intervals/rankintervals])
		
	#print rankcounts[n]
	#print pred_rankcounts[n]
	#print 
	#print str(n)+"-"*50
	#print 
	#with open("output_MAPE.txt", "a") as out:
		#out.write(dataset + "\t" + names[n] + "\t" + str(float(testMAPE)) + "\n")
		
def convert_to_ranks(l):
	aux = [(l[i], i) for i in range(len(l))]
	aux2 = sorted(aux, key=lambda x:x[0], reverse=True)
	rl = list(l)
	for i in range(len(l)):
		x = aux2[i]
		rl[x[1]] = i
	return rl
	
def SRCC(l1, l2):
	rl1 = convert_to_ranks(l1)
	rl2 = convert_to_ranks(l2)
	return np.corrcoef(rl1, rl2)[0][1]

r = []	

for x in range(rankintervals):
	ranklist = convert_to_ranks([rankcounts[n][x] for n in range(p)])
	pred_ranklist = convert_to_ranks([pred_rankcounts[n][x] for n in range(p)])
	print ranklist, pred_ranklist
	#r.append(np.corrcoef(ranklist, pred_ranklist)[0][1])
	val = SRCC([rankcounts[n][x] for n in range(p)], [pred_rankcounts[n][x] for n in range(p)])
	r.append(val)
	#print "Dataset:", dataset, "Interval:", intervals[x*(no_intervals-1)/rankintervals][1], intervals[(x+1)*(no_intervals-1)/rankintervals][1], "SRCC:", val
	
r2 = []
for x in range(last_training_interval+1,no_intervals+1):
	r2.append(SRCC([intervalcounts[n][x] for n in range(p)], [pred_intervalcounts[n][x] for n in range(p)]))

print r
meanSRCC = sum(r)/rankintervals
meanSRCC2 = sum(r2)/no_test_intervals

print "Mean MAPE:", MAPEsum/p, "Mean SRCC:", meanSRCC2, "No. of intervals:", no_intervals
