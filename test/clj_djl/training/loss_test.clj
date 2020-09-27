(ns clj-djl.training.loss-test
  (:require [clojure.test :refer :all]
            [clj-djl.ndarray :as nd]
            [clj-djl.training.loss :as loss])
  (:import [ai.djl.ndarray.types Shape DataType]
           [ai.djl.ndarray NDList NDArrays]))

(deftest l1-loss
  (testing "l1-loss test."
    (with-test
      (def manager (nd/new-base-manager))
      (def pred  (nd/create manager (float-array [1 2 3 4 5])))
      (def label (nd/ones manager [5]))
      (is (= 2. (-> (loss/l1-loss) (loss/evaluate label pred) (nd/get-element)))))))
