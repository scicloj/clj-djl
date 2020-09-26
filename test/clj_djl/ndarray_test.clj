(ns clj-djl.ndarray-test
  (:require [clojure.test :refer :all]
            [clj-djl.ndarray :as nd])
  (:import [ai.djl.ndarray.types Shape DataType]))

(deftest ndarray-creation
  (testing "ndarray/create."
    (with-test
      (def manager (nd/new-base-manager))
      ;; test scalar
      (def ndarray (nd/create manager -100.))
      (is (= -100. (nd/get-element ndarray [])))
      (is (= (nd/new-shape []) (nd/get-shape ndarray)))
      (is (nd/scalar? ndarray))
      ;; test zero-dim
      (def ndarray (nd/create manager (float-array []) (nd/new-shape [1 0])))
      (is (= (nd/new-shape [1 0]) (nd/get-shape ndarray)))
      (is (= 0 (count (nd/to-array ndarray))))
      )))

(deftest size-test
  (testing "ndarray/size."
    (is (= 0 (nd/size (nd/arange 0 0))))
    (is (= 100
           (nd/size (nd/arange 0 100))
           (nd/size (nd/reshape (nd/arange 0 100) [10 10]))
           (nd/size (nd/zeros [10 10]))
           (nd/size (nd/ones [10 10]))))))
