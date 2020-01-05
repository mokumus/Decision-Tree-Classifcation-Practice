import pandas as pd
import numpy as np # linear algebra

def remove_garbage(df):
	df.dropna(inplace=True)
	df = df.loc[:, [i for i in df.columns]]
	indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
	return df[indices_to_keep].astype(np.float64)

# Calculate the Gini index for a split dataset
def gini_index(groups, classes):
	# count all samples at split point
	n_instances = float(sum([len(group) for group in groups]))
	# sum weighted Gini index for each group
	gini = 0.0
	for group in groups:
		size = float(len(group))
		# avoid divide by zero
		if size == 0:
			continue
		score = 0.0
		# score the group based on the score for each class
		for class_val in classes:
			p = [row[-1] for row in group].count(class_val) / size
			score += p * p
		# weight the group score by its relative size
		gini += (1.0 - score) * (size / n_instances)
	return gini

# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right

# Select the best split point for a dataset
def get_split(dataset):
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 10000, 10000, 10000, None
	for index in range(len(dataset[0])-1):
		for row in dataset:
			groups = test_split(index, row[index], dataset)
			gini = gini_index(groups, class_values)
			print('X%d < %.3f Gini=%.3f' % ((index+1), row[index], gini))
			if gini < b_score:
				b_index, b_value, b_score, b_groups = index, row[index], gini, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}

# Create a terminal node value
def to_terminal(group):
	outcomes = [row[-1] for row in group]
	return max(set(outcomes), key=outcomes.count)

# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
	left, right = node['groups']
	del(node['groups'])
	# check for a no split
	if not left or not right:
		node['left'] = node['right'] = to_terminal(left + right)
		return
	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = to_terminal(left), to_terminal(right)
		return
	# process left child
	if len(left) <= min_size:
		node['left'] = to_terminal(left)
	else:
		node['left'] = get_split(left)
		split(node['left'], max_depth, min_size, depth+1)
	# process right child
	if len(right) <= min_size:
		node['right'] = to_terminal(right)
	else:
		node['right'] = get_split(right)
		split(node['right'], max_depth, min_size, depth+1)

# Build a decision tree
def build_tree(train, max_depth, min_size):
	root = get_split(train)
	split(root, max_depth, min_size, 1)
	return root

# Print a decision tree
def print_tree(node, depth=0):
	if isinstance(node, dict):
		print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
		print_tree(node['left'], depth+1)
		print_tree(node['right'], depth+1)
	else:
		print('%s[%s]' % ((depth*' ', node)))

def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']

# Classification and Regression Tree Algorithm
def decision_tree(train, test, max_depth, min_size):
	tree = build_tree(train, max_depth, min_size)
	predictions = list()
	for row in test:
		prediction = predict(tree, row)
		predictions.append(prediction)
	return(predictions)


def main() :
	#Set display option for data frames
	pd.set_option('display.max_columns', 11)
	pd.set_option('display.width', 200)

	#Read data and remove garbage
	df = pd.read_csv('winequalityN.csv')
	df = remove_garbage(pd.DataFrame(data=df, columns=list(df.columns.values)))
	cols = df.columns.tolist()
	cols = cols[1:] +cols[0:1]
	df = df[cols]

	#Extract training data by white/red wine, sample size 250
	df_white = df[(df['type'] == 0.0)]
	df_red = df[(df['type'] == 1.0)]

	df_white_training = df_white.sample(n=10, random_state=1)
	df_red_training = df_red.sample(n=10, random_state=1)
	tmp_frames  = [df_white_training, df_red_training]
	df_training = pd.concat(tmp_frames)


	#gi = gini_index([df.values[1:20], df.values[-60:-40]] , df.values[1])
	#print(gi)

	print(df_training.describe())

	tree = build_tree(df_training.values, 3, 3)
	print_tree(tree)


	# evaluate algorithm
	n_folds = 5
	max_depth = 5
	min_size = 10
	dt = decision_tree(df_training.values, df_white.values,max_depth,min_size)
	dt2 = decision_tree(df_training.values, df.values, max_depth, min_size)
	print(dt2)


main()
