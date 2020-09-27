(ns clj-djl.ndarray-test
  (:require [clojure.test :refer :all]
            [clj-djl.ndarray :as nd])
  (:import [ai.djl.ndarray.types Shape DataType]
           [ai.djl.ndarray NDList NDArrays]))

(deftest creation
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
      ;; test 1d
      (def data (-> (range 0 100) (double-array)))
      (def ndarray (nd/create manager data))
      (is (= ndarray (nd/arange manager 0 100 1 "float64" (nd/get-device ndarray))))
      (is (= ndarray (nd/arange manager 0 100 1 "FLOAT64" (nd/get-device ndarray))))
      (is (= ndarray (nd/arange manager 0 100 1 DataType/FLOAT64 (nd/get-device ndarray))))
      ;; test 2d
      (def data2D (into-array [data data]))
      (def ndarray (nd/create manager data2D))
      (is (= ndarray (nd/stack [(nd/create manager data) (nd/create manager data)])))
      ;; test boolean
      (def ndarray (nd/create manager (boolean-array [true false true false])
                              (nd/new-shape [2 2])))
      (def expected (nd/create manager (int-array [1 0 1 0]) (nd/new-shape [2 2])))
      (is (= (nd/to-type ndarray "int32" false) expected))
      (is (= ndarray (nd/to-type expected "boolean" false)))
      (is (= (nd/to-type ndarray "INT32" false) expected))
      (is (= ndarray (nd/to-type expected "BOOLEAN" false)))
      (is (= (nd/to-type ndarray DataType/INT32 false) expected))
      (is (= ndarray (nd/to-type expected DataType/BOOLEAN false))))))



(deftest size-test
  (testing "ndarray/size."
    (is (= 0 (nd/size (nd/arange manager 0 0))))
    (is (= 100
           (nd/size (nd/arange manager 0 100))
           (nd/size (nd/reshape (nd/arange manager 0 100) [10 10]))
           (nd/size (nd/zeros manager [10 10]))
           (nd/size (nd/ones manager [10 10]))))))
