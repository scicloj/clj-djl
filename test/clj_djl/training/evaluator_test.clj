(ns clj-djl.training.evaluator-test
  (:require [clojure.test :refer :all]
            [clj-djl.ndarray :as nd]
            [clj-djl.training :as t]))

(deftest accuracy
  (with-open [ndm (nd/new-base-manager)]
    (let [predictions (nd/create ndm [0.3, 0.7, 0., 1., 0.4, 0.6] [3 2])
          labels (nd/create ndm [0 1 1] [3])
          acc (-> (t/new-accuracy)
                  (t/add-accumulator "")
                  (t/update-accumulator "" (nd/ndlist labels) (nd/ndlist predictions)))
          accuracy (t/get-accumulator acc "")
          expected (/ 2. 3.)]
      ;; equal within tolerance 0.00001, maybe there are better solution
      (is (< (Math/abs (- accuracy expected)) 0.00001)))))

(deftest topk-accuracy
  (with-open [ndm (nd/new-base-manager)]
    (let [predictions (nd/create ndm [0.1  0.2  0.3  0.4
                                      0    1    0    0
                                      0.3  0.5  0.1  0.1]
                                 [3 4])
          labels (nd/create ndm [0 1 2] [3])
          topk (-> (t/new-topk-accuracy 2)
                   (t/add-accumulator "")
                   (t/update-accumulator "" (nd/ndlist labels) (nd/ndlist predictions)))
          expected (/ 1. 3.)
          accuracy (t/get-accumulator topk "")]
      (is (< (Math/abs (- accuracy expected)) 0.00001)))))
