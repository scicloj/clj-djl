(ns clj-djl.ndarray.ndarray-creation-test
  (:require [clojure.test :refer :all]
            [clj-djl.ndarray :as nd]
            [clj-djl.engine :as engine]
            [clj-djl.utils :refer :all]
            [clojure.core.matrix :as matrix])
  (:import [ai.djl.ndarray.types DataType]
           [java.nio FloatBuffer]))


(deftest size-test
  (testing "ndarray/size."
    (def ndm (nd/new-base-manager))
    (is (= 0 (nd/size (nd/arange ndm 0 0))))
    (is (= 100
           (nd/size (nd/arange ndm 0 100))
           (nd/size (nd/reshape (nd/arange ndm 0 100) [10 10]))
           (nd/size (nd/zeros ndm [10 10]))
           (nd/size (nd/ones ndm [10 10]))))))

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
    (let [expected [1. 2. 3. 4. 5. 6.]
          buf (FloatBuffer/wrap (float-array expected))
          indices (long-array [0 1 3])
          nd (nd/create-row-sparse ndm buf (nd/shape 3 2) indices (nd/shape 4 2))
          array (nd/to-array nd)]
      (is (= (aget array 0) (expected 0)))
      (is (= (aget array 1) (expected 1)))
      (is (= (aget array 2) (expected 2)))
      (is (= (aget array 3) (expected 3)))
      (is (= (aget array 6) (expected 4)))
      (is (= (aget array 7) (expected 5)))
      (is (nd/is-sparse nd)))
    (let [expected [1 2 3 4 5 6]
          indices [0 1 3]
          nd (nd/create-row-sparse ndm expected (nd/shape 3 2) indices (nd/shape 4 2))
          array (nd/to-array nd)]
      (is (= (aget array 0) (expected 0)))
      (is (= (aget array 1) (expected 1)))
      (is (= (aget array 2) (expected 2)))
      (is (= (aget array 3) (expected 3)))
      (is (= (aget array 6) (expected 4)))
      (is (= (aget array 7) (expected 5)))
      (is (nd/is-sparse nd)))
    (let [expected [1 2 3 4 5 6]
          indices [0 1 3]
          nd (nd/create-row-sparse ndm expected [3 2] indices [4 2])
          array (nd/to-array nd)]
      (is (= (aget array 0) (expected 0)))
      (is (= (aget array 1) (expected 1)))
      (is (= (aget array 2) (expected 2)))
      (is (= (aget array 3) (expected 3)))
      (is (= (aget array 6) (expected 4)))
      (is (= (aget array 7) (expected 5)))
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
    (let [array (nd/zeros ndm [2 3])
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
          expected (nd/create ndm (nd/shape [0 1]))
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

(deftest full-test
  (with-open [ndm (nd/new-base-manager)]
    (let [array (nd/full ndm (nd/shape 5) (int 3))
          expected (nd/create ndm (int-array (repeat 5 3)))]
      (is (= array expected)))
    (let [array (nd/full ndm [5] 3)
          expected (nd/create ndm (int-array (repeat 5 3)))]
      (is (= array expected)))
    (let [array (nd/full ndm 5 3)
          expected (nd/create ndm (int-array (repeat 5 3)))]
      (is (= array expected)))

    (let [array (nd/full ndm (nd/shape 6) (float 5))
          expected (nd/create ndm (float-array (repeat 6 5)))]
      (is (= array expected)))
    (let [array (nd/full ndm 6 5.)
          expected (nd/create ndm (float-array (repeat 6 5)))]
      (is (= array expected)))
    ;; multi-dim
    (let [array (nd/full ndm (nd/shape 2 3) (int -100))
          expected (nd/create ndm (int-array (repeat 6 -100)) (nd/shape 2 3))]
      (is (= array expected)))
    (let [array (nd/full ndm [2 3] -100)
          expected (nd/create ndm (int-array (repeat 6 -100)) (nd/shape 2 3))]
      (is (= array expected)))
    (let [array (nd/full ndm (nd/shape 3 2) (float 4))
          expected (nd/create ndm (float-array (repeat 6 4)) (nd/shape 3 2))]
      (is (= array expected)))
    (let [array (nd/full ndm [3 2] 4.)
          expected (nd/create ndm (float-array (repeat 6 4)) (nd/shape 3 2))]
      (is (= array expected)))
    ;; scalar
    (let [array (nd/full ndm (nd/shape) (float 1))
          expected (nd/create ndm (float 1))]
      (is (= array expected)))
    (let [array (nd/full ndm [] 1.)
          expected (nd/create ndm (float 1))]
      (is (= array expected)))
    (let [array (nd/full ndm (nd/shape) (int 0))
          expected (nd/create ndm (int 0))]
      (is (= array expected)))
    (let [array (nd/full ndm [] 0)
          expected (nd/create ndm (int 0))]
      (is (= array expected)))
    ;; zero-dim
    (let [array (nd/ones ndm (nd/shape 0 1))
          expected (nd/create ndm (nd/shape 0 1))]
      (is (= array expected)))
    (let [array (nd/ones ndm [0 1])
          expected (nd/create ndm (nd/shape 0 1))]
      (is (= array expected)))))

(deftest eye-test
  (with-open [ndm (nd/new-base-manager)]
    (let [array (nd/eye ndm 2)
          expected (nd/create ndm (float-array [1 0 0 1]) (nd/shape 2 2))]
      (is (= array expected)))
    (let [array (nd/eye ndm 2 3 0)
          expected (nd/create ndm (float-array [1 0 0 0 1 0]) (nd/shape 2 3))]
      (is (= array expected)))
    (let [array (nd/eye ndm 3 4 0)
          expected (nd/create ndm (float-array [1 0 0 0, 0 1 0 0, 0 0 1 0]) (nd/shape 3 4))]
      (is (= array expected)))))

(deftest linspace-test
  (with-open [ndm (nd/new-base-manager)]
    (let [array (nd/linspace ndm 0. 9. 10 true (nd/get-device ndm))
          expected (nd/arange ndm 10.)]
      (is (= array expected)))
    (let [array (nd/linspace ndm 0. 10. 10 false (nd/get-device ndm))
          expected (nd/arange ndm 10.)]
      (is (= array expected)))
    (let [array (nd/linspace ndm 10 0 10 false (nd/get-device ndm))
          expected (nd/create ndm (float-array (range 10 0 -1)))]
      (is (= array expected)))
    (let [array (nd/linspace ndm 10 10 10)
          expected (nd/* (nd/ones ndm [10]) 10)]
      (is (= array expected)))
    ;; corner case
    (let [array (nd/linspace ndm 0 10 0)
          expected (nd/create ndm (nd/shape 0))]
      (is (= array expected)))))


#_(deftest random-integer-test
  (let [test-cases {0 2, 1000000 2000000, -1234567 -1234567}]
    (with-open [ndm (nd/new-base-manager)]
      (for [[low high] test-cases]
        (let [rand-long (nd/random-integer ndm low high (nd/shape 100 100) :int64)
              mean (-> (nd/to-type rand-long :float64 false)
                       (nd/mean)
                       (nd/get-element))
              max (-> (nd/max rand-long)
                      (nd/get-element))
              min (-> (nd/min rand-long)
                      (nd/get-element))]
          (is (< max high))
          (is (>= min low))
          (is (and (>= mean low)
                   (< mean high))))))))

(deftest random-uniform-test
  (with-open [ndm (nd/new-base-manager)]
    (let [uniform (nd/random-uniform ndm 0 10 [1000 1000])]
      (is (>= (nd/get-element (nd/min uniform)) 0))
      (is (< (nd/get-element (nd/min uniform)) 10))
      (is (< (Math/abs (- (nd/get-element (nd/mean uniform)) 5.)) 0.01)))))

(deftest random-normal-test
  (with-open [ndm (nd/new-base-manager)]
    (let [normal (nd/random-normal ndm [1000 1000])
          mean (nd/mean normal)
          std (-> normal (nd/- mean) (nd/pow 2) (nd/mean))]
      (is (< (Math/abs (- (nd/get-element mean) 0.)) 0.01))
      (is (< (Math/abs (- (nd/get-element std) 1.)) 0.01)))))

(deftest fixed-seed-test
  (with-open [ndm (nd/new-base-manager)]
    (if-not (= "TensorFlow" (engine/get-engine-name (engine/get-instance)))
      (let [fixed-seed 1234]
        (let [_ (.setRandomSeed (engine/get-instance) fixed-seed)
              expected-uniform (nd/random-uniform ndm -10 10 [10 10])
              _ (.setRandomSeed (engine/get-instance) fixed-seed)
              actual-uniform (nd/random-uniform ndm -10 10 [10 10])]
          (is (< (Math/abs (nd/get-element (nd/sum (nd/- expected-uniform actual-uniform))))
                 0.001)))
        (let [_ (.setRandomSeed (engine/get-instance) fixed-seed)
              expected-normal (nd/random-normal ndm [10 10])
              _ (.setRandomSeed (engine/get-instance) fixed-seed)
              actual-normal (nd/random-normal ndm [10 10])]
          (is (< (Math/abs (nd/get-element (nd/sum (nd/- expected-normal actual-normal))))
                 0.001)))))))
