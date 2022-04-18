import numpy as np
import scipy as sp
import pandas as pd
import sklearn.linear_model
import sklearn.ensemble
import sklearn.model_selection
import itertools as it

class Util:
    
    def __init__(self, dataset_meta_features, validation_learning_curves, test_learning_curves):
        
        self.dsmf = dataset_meta_features
        
        self.datasets = sorted(validation_learning_curves.keys())
        self.algorithms = sorted(validation_learning_curves[self.datasets[0]])
        self.validation_learning_curves = validation_learning_curves
        self.test_learning_curves = test_learning_curves
        
        self.performance_models = None
        self.time_models = None
    
    def reset_dataset(self, dataset_meta_features):
        
        # dataset descriptors
        self.dataset_meta_features = dataset_meta_features
        self.time_budget = int(dataset_meta_features["time_budget"])
        
        # bookkeeping variables
        self.remaining_budget = self.time_budget
        self.observations = []
        self.actions = []
        self.wasted_budgets = []
        self.curves = [([], []) for i, a in enumerate(self.algorithms)] # create ds to store observed learning curves
        self.choice = None
        self.best_score = -np.inf
        self.last_score_of_choice = -np.inf
        self.iterations_without_new_anchor = {i: 0 for i in range(len(self.algorithms))}
        self.times_to_first_true_observed_value = {}
        
        # create predictions of scores
        if self.performance_models is not None:
            scores = self.predict_alc_of_all_algorithms(dataset_meta_features)
            self.preferences_alc = np.flip(np.argsort(scores))
            best_anticipated_result = max(scores)
        else:
            self.preferences_alc = None
        
        # create predictions of times to first solution
        if self.time_models is not None:
            self.anticipiated_times_to_first_result = self.predict_time_to_first_result_of_all_algorithms(dataset_meta_features)
        else:
            self.anticipiated_times_to_first_result = None
    
    
    def register_observation(self, observation):
        
        if len(self.observations) + 1 != len(self.actions):
            raise Exception("Number of actions would not equal number of observations!\n\t" + str(len(self.observations)) + " observations so far: " + str(self.observations) + "\n\t" + str(len(self.actions)) + " actions so far:" + str(self.actions))
        
        self.observations.append(observation)
        new_value_observed = False
        
        # add observation to learning curve
        if observation is not None:
            
            # was a new anchor observed for the explored algorithm?
            current_curve = self.curves[observation[0]]
            highest_timestamp = current_curve[0][-1] if len(current_curve[0]) > 0 else 0
            
            if observation[1] > highest_timestamp:
                current_curve[0].append(observation[1])
                current_curve[1].append(observation[2])
                self.iterations_without_new_anchor[observation[0]] = 0
                new_value_observed = True
                
                # if this is the first timestamp, memorize it
                if highest_timestamp == 0:
                    self.times_to_first_true_observed_value[observation[0]] = observation[1]
                
            else:
                self.iterations_without_new_anchor[observation[0]] += 1
            
            
            # if the tried algorithm was the choice, memorize the last best one
            if observation[0] == self.choice:
                self.last_score_of_choice = observation[2]
            
            # update best choice
            if observation[2] > self.last_score_of_choice:
                self.best_score = observation[2]
                self.choice = observation[0]
                
            # now do register the observation
            history_indices_for_last_algorithm = [i for i, o in enumerate(self.observations) if o[0] == observation[0]]
            actions_for_algorithms = [self.actions[i] for i in history_indices_for_last_algorithm]
            observations_for_algorithm = [self.observations[i] for i in history_indices_for_last_algorithm]
            
            advancement_on_curve = observations_for_algorithm[-1][1] - observations_for_algorithm[-2][1] if len(observations_for_algorithm) > 1 else observations_for_algorithm[-1][1]
            invested_budget = actions_for_algorithms[-1][2]
            wasted_budget = np.round(invested_budget - advancement_on_curve, 2)
            self.wasted_budgets.append(wasted_budget)
            self.wasted_budget_absolute = wasted_budget
            self.wasted_budget_relative = wasted_budget_relative = np.round(100 * sum(self.wasted_budgets) / self.time_budget, 2)
            
            MAX_BUDGET_WASTE = 100
            if wasted_budget_relative > MAX_BUDGET_WASTE:
                raise Exception(f"Wasted more than {MAX_BUDGET_WASTE}% of the budget!")
                
            
        return new_value_observed
    
    def tell_action(self, action):
        if len(self.observations) != len(self.actions):
            raise Exception("Cannot insert ACTION " + str(action) + ". Number of actions is not equal number of observations!\n\t" + str(len(self.observations)) + " observations so far: " + str(self.observations) + "\n\t" + str(len(self.actions)) + " actions so far:" + str(self.actions))
        
        self.actions.append(action)
        self.remaining_budget -= action[2]
        
        
        
    def is_lc_stale(self, algo):
        slopes = self.get_slopes(algo)
        slope_in_last_passage = self.get_slope_since_last_k_anchor(algo, 2)
        return len(slopes) > 2 and (max(slopes[-2:]) < 10**-5 or slope_in_last_passage < 10**-5)
        
    
    def get_current_alc(self):
        curve = []
        t = 0
        times_in_algos = {a: 0 for a in self.algorithms}
        for action, observation in zip(self.actions, [None] + self.observations):
            t += action[2]
            choice = action[0]
            tested = action[1]
            times_in_algos[self.algorithms[tested]] += action[2]
            if choice is not None:
                time_in_choice = times_in_algos[self.algorithms[choice]]
                curve_of_choice = np.array(self.curves[choice])
                indices = np.where(curve_of_choice[0] <= time_in_choice)[0]
                if len(indices) > 0:
                    max_anchor_in_choice = max(indices)
                
                    curve.append([t, curve_of_choice[1,max_anchor_in_choice]])
        if len(curve) > 0:
            curve = np.array(curve).T
            return self.get_alc(curve[0], curve[1], self.time_budget)
        else:
            return 0


    '''
        Computes the alc for a concrete learning curve
    '''
    def get_alc(self, timestamps, scores, time_budget, start = None, end = None):
        start = 0 if start is None else start / time_budget
        end = 1 if end is None else end / time_budget
        timestamps_normalized = np.array(timestamps) / time_budget
        alc = 0.0
        for i, t in enumerate(timestamps_normalized):
            if t >= start and t <= end:
                if i==0:
                    alc += scores[i] * (1-t)
                else:
                    alc += (scores[i] - scores[i-1]) * (1-t)
        return alc
    
    
    '''
        computes the alc values of all algorithms on all datasets.
        
        the alc matrix (rows for datasets, cols for algorithms) will be stored in self.alc_valid and self.alc_test
        
        it also computes several statistics for the algorithms by aggregating over this matrix.
        
    '''
    def compute_alcs(self):
        
        # compute basic alcs
        self.alc_valid = np.zeros((len(self.datasets), len(self.algorithms)))
        self.alc_test = np.zeros((len(self.datasets), len(self.algorithms)))
        for i, dataset_name in enumerate(self.datasets):
            time_budget = int(self.dsmf[dataset_name]["time_budget"])
            for j, algo in enumerate(self.algorithms):
                self.alc_valid[i,j] = self.get_alc(self.validation_learning_curves[dataset_name][algo].timestamps, self.validation_learning_curves[dataset_name][algo].scores, time_budget)
                self.alc_test[i,j] = self.get_alc(self.test_learning_curves[dataset_name][algo].timestamps, self.test_learning_curves[dataset_name][algo].scores, time_budget)
        
        # compute alc regrets
        self.alc_regrets_valid = np.zeros((len(self.datasets), len(self.algorithms)))
        self.alc_regrets_test = np.zeros((len(self.datasets), len(self.algorithms)))
        for i, ds in enumerate(self.datasets):
            best_alc_valid = np.max(self.alc_valid[i])
            best_alc_test = np.max(self.alc_test[i])
            for j, algo in enumerate(self.algorithms):
                self.alc_regrets_valid[i,j] = best_alc_valid - self.alc_valid[i,j]
                self.alc_regrets_test[i,j] = best_alc_test - self.alc_test[i,j]
        
        # compute statistics
        self.mean_alcs_valid = np.mean(self.alc_valid, axis=0)
        self.median_alcs_valid = np.median(self.alc_valid, axis=0)
        self.optimistic_alcs_valid = np.percentile(self.alc_valid, 75, axis=0)
        self.pessimistic_alcs_valid = np.percentile(self.alc_valid, 25, axis=0)
        self.mean_alc_regrets_valid = np.mean(self.alc_regrets_valid, axis=0)
        self.median_alc_regrets_valid = np.median(self.alc_regrets_valid, axis=0)
        self.optimistic_alc_regrets_valid = np.percentile(self.alc_regrets_valid, 75, axis=0)
        self.pessimistic_alc_regrets_valid = np.percentile(self.alc_regrets_valid, 25, axis=0)
        
        
        self.mean_alcs_test = np.mean(self.alc_test, axis=0)
        self.median_alcs_test = np.median(self.alc_test, axis=0)
        self.optimistic_alcs_test = np.percentile(self.alc_test, 75, axis=0)
        self.pessimistic_alcs_test = np.percentile(self.alc_test, 25, axis=0)
        self.mean_alc_regrets_test = np.mean(self.alc_regrets_test, axis=0)
        self.median_alc_regrets_test = np.median(self.alc_regrets_test, axis=0)
        self.optimistic_alc_regrets_test = np.percentile(self.alc_regrets_test, 75, axis=0)
        self.pessimistic_alc_regrets_test = np.percentile(self.alc_regrets_test, 25, axis=0)
    
    
    
    '''
        Learns a RandomForestRegressor for each algorithm that will predict the ALC performance based on meta-features.
        
        This must be invoked prior to call "predict_alc"
    '''
    def learn_alc_models(self, N_ESTIMATORS = 100, report_validation_performance = False):
        
        # compute ALC values
        alc_valid = np.zeros((len(self.datasets), len(self.algorithms)))
        alc_test = np.zeros((len(self.datasets), len(self.algorithms)))
        for i, dataset_name in enumerate(self.datasets):
            for j, algo in enumerate(self.algorithms):
                alc_valid[i,j] = self.get_alc(self.validation_learning_curves[dataset_name][algo].timestamps, self.validation_learning_curves[dataset_name][algo].scores)
        
        # create trainig set
        self.performance_models = {}
        for j, algo in enumerate(self.algorithms):
            rows = []
            for i, dataset_name in enumerate(self.datasets):
                row = [int(self.dsmf[dataset_name]["train_num"]), int(self.dsmf[dataset_name]["feat_num"])]
                row.append(alc_valid[i,j])
                rows.append(row)
            df = pd.DataFrame(rows, columns=["train_num", "feat_num", "alc"])
            X = df[["train_num", "feat_num"]].values
            y = df["alc"].values
            
            # train model
            #X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y)
            rf = sklearn.ensemble.RandomForestRegressor(n_estimators = N_ESTIMATORS)
            
            # reports
            if report_validation_performance:
                for i in range(10):
                    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y)
                    rf.fit(X_train, y_train)
                    y_hat = rf.predict(X_test)
                    rows = []
                    for truth, prediction in zip(y_test, y_hat):
                        rows.append([truth, prediction, truth - prediction])
            rf.fit(X, y)
            self.performance_models[algo] = rf
    
    
    '''
        predicts the alc of an algorithm on some new datasets (described by the dataset meta features)
    '''
    def predict_alc(self, algo, dataset_meta_features):
        x_1 = [int(dataset_meta_features["train_num"]), int(dataset_meta_features["feat_num"])]
        if self.util.performance_models is None:
            raise Exception("Call learn_alc_models first!")
        return self.performance_models[algo].predict([x_1])[0]
    
    '''
        returns a list with the anticipiated alcs for all algorithms (same order as in self.algorithms)
    '''
    def predict_alc_of_all_algorithms(self, dataset_meta_features):
        return [self.predict_alc(algo, dataset_meta_features) for algo in self.algorithms]
    
            
    '''
        learn for each pair of algorithms whether i is faster than j in producing a first non-trivial solution
    '''
    def learn_faster_algo_model(self, N_ESTIMATORS = 100, quality_threshold = 0.1, report_validation_scores = False):
        
        self.faster_models = np.empty((len(self.algorithms), len(self.algorithms)), dtype=object)
        for a1, a2 in it.combinations(range(len(self.algorithms)), 2):
            
            X = []
            y = []
            for i, dataset_name in enumerate(self.datasets):
                lc1 = self.get_lc_from_threshold(dataset_name, self.algorithms[a1], quality_threshold)
                lc2 = self.get_lc_from_threshold(dataset_name, self.algorithms[a2], quality_threshold)
                
                # if both algorithms have such a performance, get the faster one
                i_is_faster = lc1[0][0] < lc2[0][0]
                X.append([int(self.dsmf[dataset_name]["train_num"]), int(self.dsmf[dataset_name]["feat_num"])])
                y.append(i_is_faster)
                
            # train model
            rf = sklearn.ensemble.RandomForestClassifier(n_estimators = N_ESTIMATORS)
            
            if report_validation_scores:
                for i in range(10):
                    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y)
                    rf.fit(X_train, y_train)
                    y_hat = rf.predict(X_test)
                    rows = []
                    for truth, prediction in zip(y_test, y_hat):
                        rows.append([truth, prediction, truth == prediction])

            rf.fit(X, y)
            self.faster_models[a1,a2] = rf
            
            
    '''
        Learns a RandomForestRegressor for each algorithm to predict the runtime until a first result will be available.
    '''
    def learn_time_models(self, N_ESTIMATORS = 100, quality_threshold = 0.02, report_validation_performance = False, learn_exponent = True):
        
        # build model that predicts when the first result will arrive on a given dataset
        self.time_models = {}
        for j, algo in enumerate(self.algorithms):
            rows = []
            for i, dataset_name in enumerate(self.datasets):
                row = [
                    int(self.dsmf[dataset_name]["train_num"]),
                    int(self.dsmf[dataset_name]["train_num"]) ** 2,
                    int(self.dsmf[dataset_name]["feat_num"]),
                    int(self.dsmf[dataset_name]["feat_num"]) ** 2,
                    int(self.dsmf[dataset_name]["label_num"])
                 ]
                valid_indices = self.get_indices_of_lc_above_threshold(dataset_name, algo, quality_threshold)
                if len(valid_indices) > 0:
                    index_with_fist_relevant_result = valid_indices[0]
                    time_to_result = self.validation_learning_curves[dataset_name][algo].timestamps[index_with_fist_relevant_result]
                else:
                    time_to_result = int(self.dsmf[dataset_name]["time_budget"]) * 2
                
                row.append((np.log(time_to_result * 1000) / np.log(10) if time_to_result > 0 else 0) if learn_exponent else time_to_result)
                rows.append(row)
            df = pd.DataFrame(rows, columns=["train_num", "train_num2", "feat_num", "feat_num2", "numlabels", "timetoresult"])
            X = df[["train_num", "train_num2", "feat_num", "feat_num2", "numlabels"]].values
            y = df["timetoresult"].values
            
            # train model
            rf = sklearn.ensemble.RandomForestRegressor(n_estimators = N_ESTIMATORS)
            
            # reports
            if report_validation_performance:
                for i in range(10):
                    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y)
                    rf.fit(X_train, y_train)
                    y_hat = rf.predict(X_test)
                    rows = []
                    for truth, prediction in zip(y_test, y_hat):
                        rows.append([truth, prediction, truth - prediction, prediction * 0.5 >= truth, prediction >= truth, prediction * 2 >= truth, prediction * 4 >= truth])
            
            rf.fit(X, y)
            self.time_models[algo] = rf

    
    '''
        predicts the runtime to a first result for an algorithm on some new dataset described by dataset_meta_features
    '''
    def predict_runtime(self, algo, dataset_meta_features):
        if self.time_models is None:
            raise Exception("First run learn_time_models to create the models for this problem.")
            
        x_2 = [int(dataset_meta_features["train_num"]), int(dataset_meta_features["train_num"]) ** 2, int(dataset_meta_features["feat_num"]), int(dataset_meta_features["feat_num"]) ** 2, int(dataset_meta_features["label_num"])]
        
        return (10**self.time_models[algo].predict([x_2])[0]) / 1000
    
    
    '''
        returns a list with the anticipiated times to first solution for all algorithms (same order as in self.algorithms)
    '''
    def predict_time_to_first_result_of_all_algorithms(self, dataset_meta_features):
        return [self.predict_runtime(algo, dataset_meta_features) for algo in self.algorithms]
    
    def get_slope_since_last_k_anchor(self, algo, k):
        curve = self.curves[algo]
        if len(curve[0]) < k + 1:
            return []
        return (curve[1][-1] - curve[1][-k - 1]) / (curve[0][-1] - curve[0][-k - 1])
    
    
    def get_slopes(self, algo):
        curve = self.curves[algo]
        if len(curve[0]) < 2:
            return []
        return [(curve[1][-i] - curve[1][-i - 1]) / (curve[0][-i] - curve[0][-i - 1]) for i in range(len(curve[0]) - 1, 0, -1)]
    
    def get_last_slope(self, algo):
        return self.get_slopes[-1]
    
    
    
    
    def get_single_best_algo(self):
        
        # always explore the same algorithm (the one with highest mean ALC)
        sources_to_max = [
            self.mean_alcs_valid,
            self.median_alcs_valid,
            self.optimistic_alcs_valid,
            self.pessimistic_alcs_valid,
            self.optimistic_alcs_valid
        ]
        sources_to_min = [
            self.mean_alc_regrets_valid,
            self.median_alc_regrets_valid,
            self.optimistic_alc_regrets_valid,
            self.pessimistic_alc_regrets_valid
        ]
        ranks = [sp.stats.rankdata(src) for src in sources_to_max]
        ranks.extend([sp.stats.rankdata(-src) for src in sources_to_min])
        ranks = np.array(ranks)
        return np.argmax(np.mean(ranks, axis=0))
        
    
    
    
    
    
    def get_best_delta(self, algo):
        
        # now determine the step size
        curve_of_active_algo = self.curves[algo]
        is_cheap = is_expensive = self.anticipiated_times_to_first_result[algo] < 10
        is_expensive = self.anticipiated_times_to_first_result[algo] > 50
        if len(curve_of_active_algo[0]) == 0:
            
            num_fails = self.iterations_without_new_anchor[algo]
            if is_expensive:
                
                if num_fails == 0:
                    delta_t = self.anticipiated_times_to_first_result[algo]
                else:
                    delta_t = 10 ** 5
            
            else: # for cheap datasets do this here
                delta_t = 0.5 + (num_fails > 0) * 4 ** num_fails
        
        else: # a factor of x will lead to a new point that is x + 1 times as high as the last one
            
            
            if len(curve_of_active_algo[0]) == 1:
                factor = 1.9
            else:
                ratio_between_times_to_last_imps = curve_of_active_algo[0][-1] / curve_of_active_algo[0][-2]
                if len(curve_of_active_algo[0]) == 2:
                    factor = 1.5
                else:
                    delta_recommended_from_last_obs = ratio_between_times_to_last_imps - 1
                    factor = np.mean([1.05, delta_recommended_from_last_obs])
            factor += .5 * self.iterations_without_new_anchor[algo]
            delta_t = curve_of_active_algo[0][-1] * factor
        
        buffer_to_full = 0.7
        if delta_t < 0:
            raise Exception(f"Computed negative delta_t: {delta_t}")
        if self.remaining_budget > 10 and delta_t >= buffer_to_full * self.remaining_budget:
            delta_t = buffer_to_full * self.remaining_budget
        return delta_t
    
    
    
    def get_indices_of_lc_above_threshold(self, dataset_name, algo, threshold):
        return np.where(self.validation_learning_curves[dataset_name][algo].scores >= threshold)[0]
    
    
    
    
    


        
    '''
        Jonas' stuff
    '''

    def compute_best_pair_of_algorithms(self, lcdb):
        best_pair = None
        best_alc = 0
        for algo_pair in it.combinations(self.algorithms, 2):
            if best_pair == None:
                best_pair = algo_pair
            mean_alc = np.mean(np.asarray([self.get_alc(*self.get_best_performance_for_dataset_and_algorithm_pair(lcdb[ds], algo_pair)) for ds in self.datasets])) 
            if mean_alc > best_alc:
                best_pair = algo_pair
                best_alc = mean_alc
        return best_pair
        

    '''
        Computes the partial normalized alc for a concrete learning curve
    '''
    def get_partial_alc(self, timestamps, scores, start_time, end_time):
        # trim learning curve to time boundaries
        timestamp_mask = np.logical_and(np.greater_equal(timestamps, np.full_like(timestamps, fill_value=start_time)), np.less_equal(timestamps, np.full_like(timestamps, fill_value=end_time)))

        timestamps_trimmed = timestamps[timestamp_mask]
        timestamps_trimmed = np.insert(timestamps_trimmed, 0, start_time)
        timestamps_trimmed = np.append(timestamps_trimmed, end_time)
        
        scores_trimmed = scores[timestamp_mask]
        if len(scores_trimmed) == 0:
            return 0
        scores_trimmed = np.insert(scores_trimmed, 0, scores_trimmed[0])
        scores_trimmed = np.append(scores_trimmed, scores_trimmed[-1])

        # normalize timestamps
        timespan = max(timestamps_trimmed) - min(timestamps_trimmed)
        timestamps_normalized = np.array(timestamps_trimmed) / timespan
        alc = 0.0
        for i, t in enumerate(timestamps_normalized):
            if i==0:
                alc += scores_trimmed[i] * (1-t)
            else:
                alc += (scores_trimmed[i] - scores_trimmed[i-1]) * (1-t)
        return alc

    def get_best_performance_for_dataset_and_algorithm_pair(self, lcs, algos, time_budget, curve_format = "competition", verbose=False):
        if curve_format == "competition":
            curves = {a: np.column_stack((curve.timestamps, curve.scores)) for a, curve in lcs.items()}
        elif curve_format == "lcdb":
            curves = {a: np.array(lcs[a]["valid"]) for a in lcs.keys()}
        else:
            raise Exception("Unknown curve format.")
        
        curve_0 = curves[algos[0]]
        curve_1 = curves[algos[1]]
        
        best_performance = self.get_alc(curve_0[:,0], curve_0[:,1], time_budget)
        best_time = None
        
        # compute best performance if algorithm is swaped 
        for i, (t, v) in enumerate(curve_0):
            
            # compute curve that is composed from the two at the respective split point
            composed_curve = []
            curve_0_cut = curve_0[:i + 1]
            remaining_time = time_budget - curve_0_cut[-1,0]
            composed_curve.extend(list(curve_0_cut))
            curve_1_cut = curve_1[curve_1[:,0] <= remaining_time]
            best_score = max(curve_0_cut[:,1])
            for t_1, v_1 in curve_1_cut:
                cur_time = curve_0_cut[-1,0] + t_1
                best_score = max(best_score, v_1)
                composed_curve.append([cur_time, best_score])
            composed_curve = np.array(composed_curve)
            
            
            performance_if_swap = self.get_alc(composed_curve[:,0], composed_curve[:,1], time_budget)
            
            if performance_if_swap > best_performance:
                best_performance = performance_if_swap
                best_time = t
        return best_performance, best_time if best_time is not None else None

    def get_best_partners_of_algorithm(self, algo):
        datasets = self.datasets
        best_partner = None
        best_alc = 0
        scores = []
        partners = []
        for partner in self.algorithms:
            if partner == algo:
                continue
            partners.append(partner)
            alcs_with_this_partner = np.asarray([self.get_best_performance_for_dataset_and_algorithm_pair(self.validation_learning_curves[ds], [algo, partner], int(self.dsmf[ds]["time_budget"]))[0] for ds in datasets])
            scores.append([np.mean(alcs_with_this_partner), np.median(alcs_with_this_partner), np.percentile(alcs_with_this_partner, 25), np.percentile(alcs_with_this_partner, 75)])
        
        scores = np.array(scores).T
        ranks = np.array([sp.stats.rankdata(src) for src in scores])
        ranks_avg = np.mean(ranks, axis=0)
        best_partner_indices = np.argsort(ranks_avg)
        best_partners = [partners[i] for i in best_partner_indices]
        alcs = [scores[0,i] for i in best_partner_indices]
        return best_partners, alcs
    
    def get_best_partner_of_algorithm(self,algo):
        best_partners, alcs = self.get_best_partners_of_algorithm(algo)
        return best_partners[0], alcs[0]
    
    def get_best_pair_of_algorithms(self):
        datasets = self.datasets
        algorithms = self.algorithms
        best_pair = None
        best_alc = 0
        for algo in algorithms:
            best_partner, mean_alc_with_partner = self.get_best_partner_of_algorithm(algo)
            if mean_alc_with_partner > best_alc:
                best_alc = mean_alc_with_partner
                best_pair = (algo, best_partner)
        return best_pair, best_alc

class Agent():
    def __init__(self, number_of_algorithms):
        """
        Initialize the agent

        Parameters
        ----------
        number_of_algorithms : int
            The number of algorithms

        """
        self.n = number_of_algorithms
        self.step2_counter = 0
        
    def reset(self, dataset_meta_features, algorithms_meta_features):
        """
        Reset the agents' memory for a new dataset

        Parameters
        ----------
        dataset_meta_features : dict of {str : str}
            The meta-features of the dataset at hand, including:
                usage = 'AutoML challenge 2014'
                name = name of the dataset
                task = 'binary.classification', 'multiclass.classification', 'multilabel.classification', 'regression'
                target_type = 'Binary', 'Categorical', 'Numerical'
                feat_type = 'Binary', 'Categorical', 'Numerical', 'Mixed'
                metric = 'bac_metric', 'auc_metric', 'f1_metric', 'pac_metric', 'a_metric', 'r2_metric'
                time_budget = total time budget for running algorithms on the dataset
                feat_num = number of features
                target_num = number of columns of target file (one, except for multi-label problems)
                label_num = number of labels (number of unique values of the targets)
                train_num = number of training examples
                valid_num = number of validation examples
                test_num = number of test examples
                has_categorical = whether there are categorical variable (yes=1, no=0)
                has_missing = whether there are missing values (yes=1, no=0)
                is_sparse = whether this is a sparse dataset (yes=1, no=0)

        algorithms_meta_features : dict of dict of {str : str}
            The meta_features of each algorithm:
                meta_feature_0 = 1 or 0
                meta_feature_1 = 0.1, 0.2, 0.3,…, 1.0

        Examples
        ----------
        >>> dataset_meta_features
        {'usage': 'AutoML challenge 2014', 'name': 'Erik', 'task': 'regression',
        'target_type': 'Binary', 'feat_type': 'Binary', 'metric': 'f1_metric',
        'time_budget': '600', 'feat_num': '9', 'target_num': '6', 'label_num': '10',
        'train_num': '17', 'valid_num': '87', 'test_num': '72', 'has_categorical': '1',
        'has_missing': '0', 'is_sparse': '1'}

        >>> algorithms_meta_features
        {'0': {'meta_feature_0': '0', 'meta_feature_1': '0.1'},
         '1': {'meta_feature_0': '1', 'meta_feature_1': '0.2'},
         '2': {'meta_feature_0': '0', 'meta_feature_1': '0.3'},
         '3': {'meta_feature_0': '1', 'meta_feature_1': '0.4'},
         ...
         '18': {'meta_feature_0': '1', 'meta_feature_1': '0.9'},
         '19': {'meta_feature_0': '0', 'meta_feature_1': '1.0'},
         }
        """

        self.dataset_meta_features = dataset_meta_features
        self.time_budget = int(dataset_meta_features["time_budget"])
        self.remaining_budget = self.time_budget
        
        self.counter = 0
        self.phase2_active = False
        self.phase_2_counter = 0
        
        # reset util object
        self.util.reset_dataset(dataset_meta_features)
        
        
        return
        
        
        # determine potential improvements on the ACL by the algorithms
        quality_threshold = 1 / int(dataset_meta_features["label_num"]) if int(dataset_meta_features["label_num"]) != 0 else 0.05
        PARAM_EXPECTED_IMPROVEMENT_FACTOR_OF_GOOD_SOLUTION = 1.5
        index_of_most_promising_algo = np.argsort(scores)[-1]
        delta1 = times[index_of_most_promising_algo]
        imps = []
        for i, algo in enumerate(self.algorithms):
            if i != index_of_most_promising_algo:
                delta2 = times[i]
                imp = quality_threshold * (delta1 - PARAM_EXPECTED_IMPROVEMENT_FACTOR_OF_GOOD_SOLUTION * delta2)
            else:
                imp = 0
            imps.append(imp)
        imps = np.array(imps)

        # pick a prioritized algorithm
        improving_algos = np.where(imps > 0)[0]
        if len(improving_algos) > 0:
            best_discrepancy = np.inf
            best_index = None
            for i in improving_algos:
                algo = self.algorithms[i]
                discrepancy = best_anticipated_result - scores[i]
                if discrepancy < best_discrepancy:
                    best_discrepancy = discrepancy
                    best_index = i
            self.prioritized_algo = (best_index, times[i]) # also memorize the expected time to obtain a first result.
        else:
            self.prioritized_algo = None
        self.priorization_stopped = self.prioritized_algo is None
        self.preference_index = 0
        
        self.is_expensive = np.median(times) > 50        
        
        # define criterion to use as preference
        self.preferences = self.preferences_alc #np.argsort(times) # self.preferences_regretpercentiles
        
    def meta_train(self, dataset_meta_features, algorithms_meta_features, validation_learning_curves, test_learning_curves):
        """
        Start meta-training the agent with the validation and test learning curves

        Parameters
        ----------
        datasets_meta_features : dict of dict of {str: str}
            Meta-features of meta-training datasets

        algorithms_meta_features : dict of dict of {str: str}
            The meta_features of all algorithms

        validation_learning_curves : dict of dict of {int : Learning_Curve}
            VALIDATION learning curves of meta-training datasets

        test_learning_curves : dict of dict of {int : Learning_Curve}
            TEST learning curves of meta-training datasets

        Examples:
        To access the meta-features of a specific dataset:
        >>> datasets_meta_features['Erik']
        {'name':'Erik', 'time_budget':'1200', ...}

        To access the validation learning curve of Algorithm 0 on the dataset 'Erik' :

        >>> validation_learning_curves['Erik']['0']
        <learning_curve.Learning_Curve object at 0x9kwq10eb49a0>

        >>> validation_learning_curves['Erik']['0'].timestamps
        [196, 319, 334, 374, 409]

        >>> validation_learning_curves['Erik']['0'].scores
        [0.6465293662860659, 0.6465293748988077, 0.6465293748988145, 0.6465293748988159, 0.6465293748988159]
        """
                
        
        # memorize the names of datasets and algorithms
        self.validation_learning_curves = validation_learning_curves
        self.test_learning_curves = test_learning_curves
        
        # initialize the util class
        self.util = Util(dataset_meta_features, validation_learning_curves, test_learning_curves)
        self.datasets = self.util.datasets
        self.algorithms = self.util.algorithms
        
        #self.learn_faster_algo_model()
        N_ESTIMATORS = 100
        self.util.learn_time_models(N_ESTIMATORS = N_ESTIMATORS, report_validation_performance = False)
        self.util.compute_alcs()
        
        # get best pair of algorithms
        self.single_best = self.util.get_single_best_algo()
        self.alternatives = [self.algorithms.index(a) for a in self.util.get_best_partners_of_algorithm(self.algorithms[self.single_best])[0]]
        
    
    
    def suggest(self, observation):
        
        # compute time regret of last action
        if observation is not None:
            
            # update model information
            if len(self.util.actions) > len(self.util.observations):
                self.util.register_observation(observation)

            if False and self.util.remaining_budget / self.util.time_budget > 0.2 and self.util.wasted_budget_absolute > 50 and self.util.wasted_budget_relative > 30:
                raise Exception()
        
        # get action
        action = self.suggest_single_best_with_flip(observation)
        
        # tell action to utility module
        if action[2] > 0:
            self.util.tell_action(action)
        return action
    
    
    
    def suggest_single_best_with_flip(self, observation):
        
        single_best = self.suggest_single_best(observation)
        
        if not self.util.is_lc_stale(single_best[1]):
            return single_best
        
        algo = self.alternatives[self.step2_counter]
        
        if self.util.is_lc_stale(algo):
            self.step2_counter += 1
            self.step2_counter = self.step2_counter % len(self.alternatives)
            algo = self.alternatives[self.step2_counter]
        
        return (self.util.choice, algo, self.util.get_best_delta(algo))
    
    
    def suggest_single_best(self, observation):
        self.counter += 1
        
        # always explore the same algorithm (the one with highest mean ALC)
        algo_to_explore = self.single_best
        
        
        # dummy action in even rounds
        if self.counter % 2 == 0:
            return (algo_to_explore, algo_to_explore, 0)
        
        delta_t = self.util.get_best_delta(algo_to_explore)
        
        # now choose
        action = (self.util.choice, algo_to_explore, delta_t)
            
        
        # update internal knowledge about time
        self.remaining_budget -= action[2]
            
        return action
    
    def suggest_sophisticated(self, observation):
        self.counter += 1
        
        # update model information
        new_value_observed = self.register_observation(observation)
        
        # if this is an even action, just confirm the last one
        if self.counter % 2 == 0:
            if self.phase2_active and new_value_observed:
                self.phase_2_counter += 1
            return (self.choice, observation[0], 0)
        
        # determine algorithm to explore next
        min_cheap_results = 1
        if len(self.times_to_first_value) < min_cheap_results: # as long as there are few observation yet, take the one believed to be the fastest
            algos_ordered_by_anticipated_time = np.argsort(self.anticipiated_times_to_first_result)
            index_to_pick = len(self.times_to_first_value)
            if self.counter % 2 == 0:
                index_to_pick -= 1
            algo_to_explore = algos_ordered_by_anticipated_time[index_to_pick]
            
        else:
            self.phase2_active = True
            counter_based_index = self.phase_2_counter
            if counter_based_index <= 19:
                algo_to_explore = self.preferences[counter_based_index]
            else:
                algo_to_explore = self.choice
        
        BASIS = 2
        
        # now determine the step size
        min_exp = 6 if self.is_expensive else 1 # default minimum exponent. Is potentially overwritten for empty learning curves
        curve_of_active_algo = self.curves[algo_to_explore]
        if len(curve_of_active_algo[0]) == 0:
            
            min_exp_based_on_estimate = max(2, np.log(self.anticipiated_times_to_first_result[algo_to_explore] * 1.5) / np.log(BASIS))
            min_exp_based_on_observations = np.log(np.percentile(list(self.times_to_first_value.values()), 75)) / np.log(BASIS) if len(self.times_to_first_value) > 0 else None
            use_observations = len(self.times_to_first_value) >= 1
            
            min_exp = min_exp_based_on_observations if use_observations else min_exp_based_on_estimate
        delta_t = BASIS ** (min_exp + self.iterations_without_new_anchor[algo_to_explore]) ## powers of 2, at least 64
        
        if self.counter == 1 and delta_t >= 0.8 * int(self.dataset_meta_features["time_budget"]):
            delta_t = 0.8 * int(self.dataset_meta_features["time_budget"])
            
        
        
        # now choose
        action = (self.choice, algo_to_explore, delta_t)
            
        
        # update internal knowledge about time
        self.remaining_budget -= action[2]
            
        return action