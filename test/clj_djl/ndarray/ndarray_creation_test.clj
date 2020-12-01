(ns clj-djl.ndarray.ndarray-creation-test
  (:require [clojure.test :refer :all]
            [clj-djl.ndarray :as nd]
            [clj-djl.utils :refer :all]
            [clojure.core.matrix :as matrix])
  (:import [ai.djl.ndarray.types DataType]
           [java.nio FloatBuffer]))

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
    (def ndarray4 (nd/create ndm [[1 0] [1 0]]))
    (is (= ndarray1 ndarray2 ndarray3 ndarray4))))

(deftest create-csr-matrix
  (testing "create CSR matrix"
    (try-let [ndm (nd/new-base-manager)]
             (let [expected (float-array [7 8 9])
                   buf (FloatBuffer/wrap expected)
                   indptr (long-array [0 2 2 3])
                   indices (long-array [0 2 1])
                   nd (.createCSR ndm buf indptr indices (nd/shape 3 4))
                   array (.toFloatArray nd)]
               (is (= (aget array 0) (aget expected 0)))
               (is (= (aget array 2) (aget expected 1)))
               (is (= (aget array 9) (aget expected 2)))
               (is (.isSparse nd))))
    ;; generalized
    (try-let [ndm (nd/new-base-manager)]
             (let [expected [7 8 9]
                   buf [7 8 9]
                   indptr (long-array [0 2 2 3])
                   indices (long-array [0 2 1])
                   nd (nd/create-csr ndm buf indptr indices (nd/shape 3 4))
                   array (nd/to-array nd)]
               (is (= (aget array 0) (expected 0)))
               (is (= (aget array 2) (expected 1)))
               (is (= (aget array 9) (expected 2)))
               (is (nd/sparse? nd))))

    (try-let [ndm (nd/new-base-manager)]
             (let [expected [7 8 9]
                   buf [7 8 9]
                   indptr [0 2 2 3]
                   indices [0 2 1]
                   shape [3 4]
                   nd (nd/create-csr ndm buf indptr indices shape)
                   array (nd/to-array nd)]
               (is (= (aget array 0) (expected 0)))
               (is (= (aget array 2) (expected 1)))
               (is (= (aget array 9) (expected 2)))
               (is (nd/sparse? nd))))

    (try-let [ndm (nd/new-base-manager)]
             (let [expected [7 8 9]
                   nd (nd/create-csr ndm [7 8 9] [0 2 2 3] [0 2 1] [3 4])
                   array (nd/to-vec nd)]
               (is (= (array 0) (expected 0)))
               (is (= (array 2) (expected 1)))
               (is (= (array 9) (expected 2)))
               (is (nd/sparse? nd))))))

(deftest create-row-sparse-matrix
  (with-open [ndm (nd/new-base-manager)]
    (let [expected [1 2 3 4 5 6]
          buf (FloatBuffer/wrap (float-array expected))
          indices (long-array [0 1 3])
          nd (nd/create-row-sparse ndm buf (nd/shape 3 2) indices (nd/shape 4 2))
          array (nd/to-array nd)]
      (is (= (aget array 0) (expected 0)))
      (is (= (aget array 1) (expected 1)))
      (is (= (aget array 2) (expected 2)))
      (is (= (aget array 3) (expected 3)))
      (is (= (aget array 6) (expected 6)))
      (is (= (aget array 7) (expected 7)))
      (is (nd/is-sparse nd)))
    (let [expected [1 2 3 4 5 6]
          indices [0 1 3]
          nd (nd/create-row-sparse ndm expected (nd/shape 3 2) indices (nd/shape 4 2))
          array (nd/to-array nd)]
      (is (= (aget array 0) (expected 0)))
      (is (= (aget array 1) (expected 1)))
      (is (= (aget array 2) (expected 2)))
      (is (= (aget array 3) (expected 3)))
      (is (= (aget array 6) (expected 6)))
      (is (= (aget array 7) (expected 7)))
      (is (nd/is-sparse nd)))
    (let [expected [1 2 3 4 5 6]
          indices [0 1 3]
          nd (nd/create-row-sparse ndm expected [3 2] indices [4 2])
          array (nd/to-array nd)]
      (is (= (aget array 0) (expected 0)))
      (is (= (aget array 1) (expected 1)))
      (is (= (aget array 2) (expected 2)))
      (is (= (aget array 3) (expected 3)))
      (is (= (aget array 6) (expected 6)))
      (is (= (aget array 7) (expected 7)))
      (is (nd/is-sparse nd)))))

;; TODO: Coo and convert to sparce test


(deftest duplicate-test
  (with-open [ndm (nd/new-base-manager)]
    (let [array (nd/zeros ndm 5)
          expected (nd/create ndm (repeat 5 (float 0)))
          duplicate (nd/duplicate array)]
      (is (= duplicate expected))
      (is (not (identical? duplicate expected))))
    ;; multi dim
    (let [array (nd/ones ndm [2 3])
          expected (nd/create ndm (repeat 6 (float 0)) [2 3])
          duplicate (nd/dup array)]
      (is (= duplicate expected))
      (is (not (identical? duplicate expected))))
    ;; scalar
    (let [array (nd/zeros ndm [])
          expected (nd/create ndm (float 0))
          duplicate (nd/dup array)]
      (is (= duplicate expected))
      (is (not (identical? duplicate expected))))
    ;; zero dim
    (let [array (nd/zeros ndm [0 1])
          expected (nd/create ndm [0 1])
          duplicate (nd/dup array)]
      (is (= duplicate expected))
      (is (not (identical? duplicate expected))))))

(deftest zeros-test
  (with-open [ndm (nd/new-base-manager)]
    (let [array (nd/zeros ndm [5])
          expected (nd/create ndm (repeat 5 (float 0)))]
      (is (= array expected)))
    ;; multi-dim
    (let [array (nd/zeros ndm [2 3])
          expected (nd/create ndm (repeat 6 (float 0)) [2 3])]
      (is (= array expected)))
    ;; scalar
    (let [array (nd/zeros ndm [])
          expected (nd/create ndm (float 0))]
      (is (= array expected)))
    ;; zero dim
    (let [array (nd/zeros ndm [0 1])
          expected (nd/create ndm (nd/shape [0 1]))]
      (is (= array expected)))))

(deftest ones-test
  (with-open [ndm (nd/new-base-manager)]
    (let [array (nd/ones ndm [5])
          expected (nd/create ndm (repeat 5 (float 1)))]
      (is (= array expected)))
    ;; multi-dim
    (let [array (nd/ones ndm [2 3])
          expected (nd/create ndm (repeat 6 (float 1)) [2 3])]
      (is (= array expected)))
    ;; scalar
    (let [array (nd/ones ndm [])
          expected (nd/create ndm (float 1))]
      (is (= array expected)))
    ;; zero dim
    (let [array (nd/ones ndm [0 1])
          expected (nd/create ndm (nd/shape [0 1]))]
      (is (= array expected)))))

(deftest size-test
  (testing "ndarray/size."
    (def ndm (nd/new-base-manager))
    (is (= 0 (nd/size (nd/arange ndm 0 0))))
    (is (= 100
           (nd/size (nd/arange ndm 0 100))
           (nd/size (nd/reshape (nd/arange ndm 0 100) [10 10]))
           (nd/size (nd/zeros ndm [10 10]))
           (nd/size (nd/ones ndm [10 10]))))))
