(ns clj-djl.training.loss-test
  (:require [clojure.test :refer :all]
            [clj-djl.ndarray :as nd]
            [clj-djl.training.loss :as loss]))

(deftest loss
  (testing "l1-loss test."
    (with-test
      (def manager (nd/new-base-manager))
      (def pred  (nd/create manager (float-array [1 2 3 4 5])))
      (def label (nd/ones manager [5]))
      (is (= 2. (-> (loss/l1-loss) (loss/evaluate label pred) (nd/get-element)))))))

(deftest l2-loss
  (testing "l2-loss test."
    (with-test
      (def manager (nd/new-base-manager))
      (def pred  (nd/create manager (float-array [1 2 3 4 5])))
      (def label (nd/ones manager [5]))
      (is (= 3. (-> (loss/l2-loss) (loss/evaluate label pred) (nd/get-element)))))))
