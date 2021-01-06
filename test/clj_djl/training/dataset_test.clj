(ns clj-djl.training.dataset-test
  (:require [clojure.test :refer :all]
            [clj-djl.training.loss :as l]
            [clj-djl.training :as t]
            [clj-djl.training.initializer :as i]
            [clj-djl.model :as m]
            [clj-djl.device :as d]
            [clj-djl.ndarray :as nd]
            [clj-djl.training.dataset :as ds]
            [clj-djl.nn :as nn])
  (:import [ai.djl.training.dataset BatchSampler SequenceSampler RandomSampler]))

(def config (t/default-training-config {:loss (l/l2-loss)
                                        :initializer (i/ones)}))

(deftest sequence-sampler-test
  (with-open [model (m/new-model {:name "model"
                                  :block (nn/identity-block)})
              manager (m/get-ndmanager model)
              trainer (m/new-trainer model config)]
    (let [dataset (-> (ds/new-array-dataset-builder)
                      (ds/set-data (nd/arange manager 0 100 1 :int64 (nd/default-device)))
                      (ds/set-sampling (ds/batch-sampler (ds/sequence-sampler) 1 false))
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
                      (ds/set-sampling (ds/batch-sampler (ds/sequence-sampler) 1 false))
                      (ds/build))
          original (map #(nd/get-element (nd/head (ds/get-batch-data %)))
                        (t/iterate-dataset trainer dataset))]
      (is (= (count original) 10)))))

(deftest batch-sampler-test
  (with-open [model (m/new-model {:name "model"
                                  :block (nn/identity-block)})
              manager (m/get-ndmanager model)
              trainer (t/new-trainer model config)]
    (let [data (nd/arange manager 0 100 1 :int64 (d/default-device))
          dataset (-> (ds/array-dataset-builder)
                      (ds/set-data data)
                      (ds/set-sampling (ds/batch-sampler (ds/sequence-sampler) 27 false))
                      (ds/build))
          ;; convert ndarray to vec here, otherwise resource is closed
          remlist (map #(nd/to-vec (nd/singleton-or-throw (ds/get-batch-data %)))
                       (t/iterate-dataset trainer dataset))]
      (is (= (count remlist) 4))
      (is (= (first remlist) (range 0 27)))))
  (with-open [model (m/new-model {:name "model"
                                  :block (nn/identity-block)})
              manager (m/get-ndmanager model)
              trainer (t/new-trainer model config)]
    (let [data (nd/arange manager 0 100 1 :int64 (d/default-device))
          dataset (-> (ds/array-dataset-builder)
                      (ds/set-data data)
                      (ds/set-sampling (ds/batch-sampler (ds/sequence-sampler) 33 true))
                      (ds/build))
          ;; convert ndarray to vec here, otherwise resource is closed
          remlist (map #(nd/to-vec (nd/singleton-or-throw (ds/get-batch-data %)))
                       (t/iterate-dataset trainer dataset))]
      (is (= (count remlist) 3))
      (is (= (last remlist) (range 66 99)))))
  (with-open [model (m/new-model {:name "model"
                                  :block (nn/identity-block)})
              manager (m/get-ndmanager model)
              trainer (t/new-trainer model config)]
    (let [data (nd/arange manager 0 100 1 :int64 (d/default-device))
          dataset (-> (ds/array-dataset-builder)
                      (ds/set-data data)
                      (ds/set-sampling (ds/batch-sampler (ds/sequence-sampler) 101 true))
                      (ds/build))
          ;; convert ndarray to vec here, otherwise resource is closed
          remlist (map #(nd/to-vec (nd/singleton-or-throw (ds/get-batch-data %)))
                       (t/iterate-dataset trainer dataset))]
      (is (empty? remlist))))
  (with-open [model (m/new-model {:name "model"
                                  :block (nn/identity-block)})
              manager (m/get-ndmanager model)
              trainer (t/new-trainer model config)]
    (let [data (nd/arange manager 0 100 1 :int64 (d/default-device))
          dataset (-> (ds/array-dataset-builder)
                      (ds/set-data data)
                      (ds/set-sampling (ds/batch-sampler (ds/sequence-sampler) 101 false))
                      (ds/build))
          ;; convert ndarray to vec here, otherwise resource is closed
          remlist (map #(nd/to-vec (nd/singleton-or-throw (ds/get-batch-data %)))
                       (t/iterate-dataset trainer dataset))]
      (is (= 1 (count remlist)))
      (is (= 100 (count (first remlist))))
      (is (= (first remlist) (range 0 100))))))

(deftest array-dataset-test
  (with-open [model (m/new-model {:name "model"
                                  :block (nn/identity-block)})
              manager (m/get-ndmanager model)
              trainer (t/new-trainer model config)]
    (let [data (-> (nd/arange manager 200) (nd/reshape 100 2))
          label (-> (nd/arange manager 100) (nd/reshape 100))
          dataset (-> (ds/array-dataset-builder)
                      (ds/set-data data)
                      (ds/opt-labels label)
                      (ds/set-sampling 20 false)
                      (ds/build))]
      (doseq [[index batch]
              (map list (range 0 100 20) (t/iterate-dataset trainer dataset))]
        (is (= (nd/reshape (nd/arange manager (* 2 index) (+ (* 2 index) 40)) 20 2)
               (nd/singleton-or-throw (ds/get-batch-data batch))))
        (is (= (nd/reshape (nd/arange manager index (+ index 20)) 20)
               (nd/singleton-or-throw (ds/get-batch-labels batch)))))))
  (with-open [model (m/new-model {:name "model"
                                  :block (nn/identity-block)})
              manager (m/get-ndmanager model)
              trainer (t/new-trainer model config)]
    (let [data (-> (nd/arange manager 200) (nd/reshape 100 2))
          label (-> (nd/arange manager 100) (nd/reshape 100))
          dataset (-> (ds/array-dataset-builder)
                      (ds/set-data data)
                      (ds/opt-labels label)
                      (ds/set-sampling 15 false)
                      (ds/build))]
      (doseq [[index batch]
              (map list (range 0 100 15) (t/iterate-dataset trainer dataset))]
        (if (not= index 90)
          (do
            (is (= (nd/reshape (nd/arange manager (* 2 index) (+ (* 2 index) 30)) 15 2)
                   (nd/singleton-or-throw (ds/get-batch-data batch))))
            (is (= (nd/reshape (nd/arange manager index (+ index 15)) 15)
                   (nd/singleton-or-throw (ds/get-batch-labels batch)))))
          (do
            (is (= (nd/reshape (nd/arange manager (* 2 index) (+ (* 2 index) 20)) 10 2)
                   (nd/singleton-or-throw (ds/get-batch-data batch))))
            (is (= (nd/reshape (nd/arange manager index (+ index 10)) 10)
                   (nd/singleton-or-throw (ds/get-batch-labels batch)))))))))
  (with-open [model (m/new-model {:name "model"
                                  :block (nn/identity-block)})
              manager (m/get-ndmanager model)
              trainer (t/new-trainer model config)]
    (let [data (-> (nd/arange manager 200) (nd/reshape 100 2))
          data2 (-> (nd/arange manager 300) (nd/reshape 100 3))
          dataset (-> (ds/array-dataset-builder)
                      (ds/set-data data data2)
                      (ds/set-sampling 10 false)
                      (ds/build))]
      (doseq [[index batch]
              (map list (range 0 100 10) (t/iterate-dataset trainer dataset))]
        (is (= 2 (count (ds/get-batch-data batch))))
        (is (= (nd/reshape (nd/arange manager (* 2 index) (+ (* 2 index) 20)) 10 2)
               (first (ds/get-batch-data batch))))
        (is (= (nd/reshape (nd/arange manager (* 3 index) (+ (* 3 index) 30)) 10 3)
               (last (ds/get-batch-data batch))))))))

;; TODO
#_(deftest multithreading-test)

(deftest dataset-to-array
  (with-open [manager (nd/new-base-manager)]
    (let [data (nd/ones manager [5 4])
          label (nd/zeros manager [5 2])
          dataset (-> (ds/array-dataset-builder)
                      (ds/set-data data)
                      (ds/opt-labels label)
                      (ds/set-sampling 32 false)
                      (ds/build))
          converted (ds/to-apair dataset)
          result-data (converted 0)
          result-label (converted 1)]
      (is (= (alength result-data) 5))
      (is (= (alength result-label) 5))
      (is (= (alength (aget result-data 0)) 4))
      (is (= (alength (aget result-label 0)) 2))
      (is (= (aget (aget result-data 0) 0) 1.))
      (is (= (aget (aget result-label 0) 0) 0.)))))
