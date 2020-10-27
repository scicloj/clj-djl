(ns clj-djl.ndarray-test
  (:require [clojure.test :refer :all]
            [clj-djl.ndarray :as nd]
            [clojure.core.matrix :as matrix])
  (:import [ai.djl.ndarray.types DataType]))

(deftest creation
  (testing "ndarray/create."
    (def ndm (nd/new-base-manager))
    ;; test scalar
    (def ndarray (nd/create ndm -100.))
    (is (= -100. (nd/get-element ndarray [])))
    (is (= (nd/new-shape []) (nd/get-shape ndarray)))
    (is (nd/scalar? ndarray))
    ;; test zero-dim
    (def ndarray (nd/create ndm (float-array []) (nd/new-shape [1 0])))
    (is (= (nd/new-shape [1 0]) (nd/get-shape ndarray)))
    (is (= 0 (count (nd/to-array ndarray))))
    ;; test 1d
    (def data (-> (range 0 100) (double-array)))
    (def ndarray (nd/create ndm data))
    (is (= ndarray (nd/arange ndm 0 100 1 "float64" (nd/get-device ndarray))))
    (is (= ndarray (nd/arange ndm 0 100 1 "FLOAT64" (nd/get-device ndarray))))
    (is (= ndarray (nd/arange ndm 0 100 1 DataType/FLOAT64 (nd/get-device ndarray))))
    ;; test 2d
    (def data2D (into-array [data data]))
    (def ndarray (nd/create ndm data2D))
    (is (= ndarray (nd/stack [(nd/create ndm data) (nd/create ndm data)])))
    ;; test boolean
    (def ndarray (nd/create ndm (boolean-array [true false true false])
                            (nd/new-shape [2 2])))
    (def expected (nd/create ndm (int-array [1 0 1 0]) (nd/new-shape [2 2])))
    (is (= (nd/to-type ndarray "int32" false) expected))
    (is (= ndarray (nd/to-type expected "boolean" false)))
    (is (= (nd/to-type ndarray "INT32" false) expected))
    (is (= ndarray (nd/to-type expected "BOOLEAN" false)))
    (is (= (nd/to-type ndarray DataType/INT32 false) expected))
    (is (= ndarray (nd/to-type expected DataType/BOOLEAN false)))))


(deftest creation-with-vec
  (testing "ndarray/create with vector"
    (def ndm (nd/new-base-manager))
    ;; test 1-dim with 1 element
    (def ndarray (nd/create ndm [1]))
    (is (= (nd/new-shape [1]) (nd/get-shape ndarray)))
    (is (= 1 (count (nd/to-array ndarray))))
    ;; test 1d
    (def data (range 0 100))
    (def ndarray (nd/create ndm data))
    (is (= ndarray (nd/arange ndm 0 100 1 "int64" (nd/get-device ndarray))))
    (is (= ndarray (nd/arange ndm 0 100 1 "INT64" (nd/get-device ndarray))))
    (is (= ndarray (nd/arange ndm 0 100 1 DataType/INT64 (nd/get-device ndarray))))
    ;; test 2d
    (def data2D [data data])
    (def ndarray (nd/create ndm data2D))
    (is (= ndarray (nd/stack [(nd/create ndm data) (nd/create ndm data)])))
    ;; test boolean
    (def ndarray (nd/create ndm [true false true false] [2 2]))
    (def expected (nd/create ndm (int-array [1 0 1 0]) (nd/shape [2 2])))
    (is (= (nd/to-type ndarray "int32" false) expected))
    (is (= ndarray (nd/to-type expected "boolean" false)))
    (is (= (nd/to-type ndarray "INT32" false) expected))
    (is (= ndarray (nd/to-type expected "BOOLEAN" false)))
    (is (= (nd/to-type ndarray DataType/INT32 false) expected))
    (is (= ndarray (nd/to-type expected DataType/BOOLEAN false)))

    (def ndarray1 (nd/create ndm (map int [1 0 1 0]) [2 2]))
    (def ndarray2 (nd/create ndm (int-array [1 0 1 0]) [2 2]))
    (def ndarray3 (nd/create ndm (int-array [1 0 1 0]) (nd/shape [2 2])))
    (is (= ndarray1 ndarray2 ndarray3))

    (def ndarray1 (nd/create ndm [1 0 1 0] [2 2]))
    (def ndarray2 (nd/create ndm (long-array [1 0 1 0]) [2 2]))
    (def ndarray3 (nd/create ndm (long-array [1 0 1 0]) (nd/shape [2 2])))
    (is (= ndarray1 ndarray2 ndarray3))))

(deftest size-test
  (testing "ndarray/size."
    (def ndm (nd/new-base-manager))
    (is (= 0 (nd/size (nd/arange ndm 0 0))))
    (is (= 100
           (nd/size (nd/arange ndm 0 100))
           (nd/size (nd/reshape (nd/arange ndm 0 100) [10 10]))
           (nd/size (nd/zeros ndm [10 10]))
           (nd/size (nd/ones ndm [10 10]))))))
