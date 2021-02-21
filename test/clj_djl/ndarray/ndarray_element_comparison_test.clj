(ns clj-djl.ndarray.ndarray-element-comparison-test
  (:require [clojure.test :refer :all]
            [clj-djl.ndarray :as nd]
            [clj-djl.utils :refer :all]))

(deftest content-equal-test
  (with-open [ndm (nd/new-base-manager)]
    (let [array1 (nd/create ndm (float-array [1 2]))
          array2 (nd/create ndm (float-array [1 2]))]
      (is (= array1 array2))
      (is (.contentEquals array1 array2)))
    (let [array1 (nd/create ndm [1 2])
          array2 (nd/create ndm [1 2])]
      (is (= array1 array2))
      (is (.contentEquals array1 array2)))
    (let [array1 (nd/ones ndm (nd/shape 2 3))
          array2 (nd/ones ndm (nd/shape 1 3))]
      (is (not (= array1 array2)))
      (is (not (.contentEquals array1 array2))))
    (let [array1 (nd/ones ndm [2 3])
          array2 (nd/ones ndm [1 3])]
      (is (not (= array1 array2)))
      (is (not (.contentEquals array1 array2))))
    ;; scalar
    (let [array1 (nd/create ndm 5.)
          array2 (nd/create ndm 5.)]
      (is (= array1 array2))
      (is (.contentEquals array1 array2)))
    (let [array1 (nd/create ndm 3)
          array2 (nd/create ndm 4)]
      (is (not (= array1 array2)))
      (is (not (.contentEquals array1 array2))))
    ;; different data type
    (let [array1 (nd/create ndm 4.)
          array2 (nd/create ndm 4)]
      (is (not (= array1 array2)))
      (is (not (.contentEquals array1 array2))))
    ;; zero dim vs zero dim
    (let [array1 (nd/create ndm (nd/shape 4 0))
          array2 (nd/create ndm (nd/shape 4 0))]
      (is (= array1 array2))
      (is (.contentEquals array1 array2)))
    (let [array1 (nd/create ndm (nd/shape 0 0 0))
          array2 (nd/create ndm (nd/shape 2 0 0))]
      (is (not (= array1 array2)))
      (is (not (.contentEquals array1 array2))))))

(deftest equals-for-scalar-test
  (with-open [ndm (nd/new-base-manager)]
    (let [array1 (nd/create ndm (float-array [1 2 3 4]))
          result (nd/= array1 2)
          expected (nd/create ndm (boolean-array [false true false false]))]
      (is (= result expected))
      (is (.contentEquals result expected)))
    (let [array1 (nd/ones ndm [4 5 2])
          result (nd/= array1 1)
          expected (-> (nd/ones ndm [4 5 2]) (nd/to-type :boolean false))]
      (is (= result expected)))
    (let [array1 (nd/create ndm (float-array [1 2 3 4]))
          array2 (nd/create ndm (float-array [1 3 3 4]))
          result (nd/= array1 array2)
          expected (nd/create ndm [true false true true])]
      (is (= result expected)))
    ;; test scalar
    (let [array1 (nd/create ndm 4)
          result (nd/= array1 4)
          expected (nd/create ndm true)]
      (is (= result expected)))
    ;; test zero dim
    (let [array1 (nd/create ndm (nd/shape 4 3 2 1 0))
          array2 (nd/create ndm (nd/shape 1 0))
          result (nd/= array1 array2)
          expected (nd/create ndm (nd/shape 4 3 2 1 0) :boolean)]
      (is (= result expected)))))

(deftest equals-for-equal-ndarray
  (with-open [ndm (nd/new-base-manager)]
    (let [array1 (nd/create ndm (float-array [1 2 3 4]))
          array2 (nd/create ndm (float-array [1 2 3 4]))
          result (nd/= array1 array2)
          expected (nd/create ndm (boolean-array (repeat 4 true)))]
      (is (= result expected)))
    (let [array1 (nd/create ndm (float-array (range 1 11)) [2 5])
          array2 (-> (nd/arange ndm 10.) (nd/+ 1) (nd/reshape 2 5))
          result (nd/= array1 array2)
          expected (-> (nd/ones ndm [2 5]) (nd/to-type :boolean false))]
      (is (= result expected)))
    ;; test scalar
    (let [array1 (nd/* (nd/ones ndm [4] :float64) 5)
          array2 (nd/create ndm 5.)
          result (nd/= array1 array2)
          expected (nd/create ndm (boolean-array (repeat 4 true)))]
      (is (= result expected)))
    ;; test zero-dim
    (let [array1 (nd/create ndm (nd/shape 4 3 0))
          array2 (nd/create ndm (nd/shape 4 3 0))
          result (nd/= array1 array2)
          expected (nd/create ndm (nd/shape 4 3 0) :boolean)]
      (is (= result expected)))))
