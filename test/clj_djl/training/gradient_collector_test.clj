(ns clj-djl.training.gradient-collector-test
  (:require [clojure.test :refer :all]
            [clj-djl.model :as m]
            [clj-djl.ndarray :as nd]
            [clj-djl.training :as t]
            [clj-djl.nn :as nn]
            [clj-djl.training.loss :as l]
            [clj-djl.training.optimizer :as optimizer]
            [clj-djl.training.dataset :as dataset]
            [clj-djl.training.tracker :as tracker])
  (:import [ai.djl.training.initializer Initializer]
           [ai.djl.training.listener TrainingListener EvaluatorTrainingListener]))

(deftest autograd-test
  (with-open [model (m/new-instance "model")
              ndm (nd/new-base-manager)]
    (m/set-block model (nn/identity-block))
    (with-open [trainer (t/new-trainer model
                                       (t/default-training-config
                                        {:loss (l/l2-loss)
                                         :initializer Initializer/ONES}))
                gc (t/new-gradient-collector trainer)]
      (let [lhs (nd/create ndm (float-array [6 -9 -12 15 0 4]) [2 3])
            rhs (nd/create ndm (float-array [2 3 -4]) [3 1])
            expected (nd/create ndm (float-array [24 -54 96 60 0 -32]) [2 3])]
        (t/attach-gradient lhs)
        (let [result (nd/dot (nd/* lhs lhs) rhs)]
          (t/backward gc result)
          (let [grad (t/get-gradient lhs)]
            (is (= expected grad))
            (.close grad)
            (is (= expected (t/get-gradient lhs))))))))
  ;; simplified version
  (with-open [model (m/new-model {:name "model"
                                  :block (nn/identity-block)})
              ndm (nd/new-base-manager)
              trainer (t/new-trainer {:model model
                                      :loss (l/l2-loss)
                                      :initializer Initializer/ONES})
              gc (t/new-gradient-collector trainer)]
    (let [lhs (nd/create ndm (float-array [6 -9 -12 15 0 4]) [2 3])
          rhs (nd/create ndm (float-array [2 3 -4]) [3 1])
          expected (nd/create ndm (float-array [24 -54 96 60 0 -32]) [2 3])]
      (t/attach-gradient lhs)
      (let [result (nd/dot (nd/* lhs lhs) rhs)]
        (t/backward gc result)
        (let [grad (t/get-gradient lhs)]
          (is (= expected grad))
          (.close grad)
          (is (= expected (t/get-gradient lhs))))))))

(deftest train-test
  (let [ndata 1000
        batchsize 10
        epochs 10
        optimizer (optimizer/sgd {:tracker (tracker/fixed 0.03)})
        config (t/default-training-config {:loss (l/l2-loss)
                                           :listeners [(EvaluatorTrainingListener.)]
                                           :initializer (Initializer/ONES)
                                           :optimizer optimizer})]
    (with-open [model (m/new-model {:name "linear"
                                    :block (nn/linear-block {:units 1})})
                manager (m/get-ndmanager model)]
      (let [weight (nd/create manager (float-array [2 -3.4]) [2 1])
            bias 4.2
            data (nd/random-normal manager [ndata (nd/size weight 0)])
            label (nd/+ (nd/dot data weight) bias)
            ;; add noise
            label-with-noise (nd/+ label
                                   (nd/random-normal
                                    manager
                                    0 0.01 (nd/get-shape label) :float32 (nd/get-device manager)))
            sampling (* batchsize (count (t/get-devices config)))
            dataset (-> (dataset/new-array-dataset-builder)
                        (dataset/set-data data)
                        (dataset/opt-labels label-with-noise)
                        (dataset/set-sampling sampling false)
                        (dataset/build))
            ]
        (with-open [trainer (m/new-trainer model config)]
          (let [input-shape (nd/shape sampling (nd/size weight 0))]
            (t/initialize trainer [input-shape])
            (doseq [epoch (range epochs)]
              (t/notify-listeners trainer (fn [listner] (.onEpoch listner trainer)))
              (doseq [batch (t/iterate-dataset trainer dataset)]
                (t/train-batch trainer batch)
                (t/step trainer)
                (.close batch)))
            (let [loss-value (.getAccumulator (t/get-loss trainer) EvaluatorTrainingListener/TRAIN_EPOCH)
                  expected 0.001]
              (is (< loss-value expected)))))))))
