(ns clj-djl.training.dataset-test
  (:require [clojure.test :refer :all]
            [clj-djl.training.loss :as l]
            [clj-djl.training :as t]
            [clj-djl.training.initializer :as i]
            [clj-djl.model :as m]
            [clj-djl.ndarray :as nd]
            [clj-djl.training.dataset :as ds]
            [clj-djl.nn :as nn])
  (:import [ai.djl.training.dataset BatchSampler SequenceSampler RandomSampler]))

(def config (t/default-training-config {:loss (l/l2-loss)
                                        :initializer i/ones}))

(deftest sequence-sampler-test
  (with-open [model (m/new-model {:name "model"
                                  :block (nn/identity-block)})
              manager (m/get-ndmanager model)
              trainer (m/new-trainer model config)]
    (let [dataset (-> (ds/new-array-dataset-builder)
                      (ds/set-data (nd/arange manager 0 100 1 :int64 (nd/default-device)))
                      (ds/set-sampling (BatchSampler. (SequenceSampler.) 1 false))
                      (ds/build))
          original (map #(nd/get-element (nd/head (ds/get-batch-data %)))
                        (t/iterate-dataset trainer dataset))
          expected (range 0 100)]
      (is (= original expected)))))

(deftest random-sampler-test
  (with-open [model (m/new-model {:name "model"
                                  :block (nn/identity-block)})
              manager (m/get-ndmanager model)
              trainer (m/new-trainer model config)]
    (let [dataset (-> (ds/new-array-dataset-builder)
                      (ds/set-data (nd/arange manager 0 10 1 :int64 (nd/default-device)))
                      (ds/set-sampling (BatchSampler. (RandomSampler.) 1 false))
                      (ds/build))
          original (map #(nd/get-element (nd/head (ds/get-batch-data %)))
                        (t/iterate-dataset trainer dataset))]
      (is (= (count original) 10)))))
