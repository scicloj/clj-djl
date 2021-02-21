(ns clj-djl.ndarray.ndarray-reduction-test
  (:require [clojure.test :refer :all]
            [clj-djl.ndarray :as nd]
            [clj-djl.utils :refer :all]
            [clojure.core.matrix :as matrix]))

(deftest max-test
  (with-open [ndm (nd/new-base-manager)]
    (let [array (nd/create ndm (float-array [1 2 5 1]))]
      (is (= (nd/get-element (nd/max array)) 5.)))
    (let [array (nd/create ndm (float-array [2 4 6 8]) (nd/shape 2 2))]
      (is (= (nd/get-element (nd/max array)) 8.))
      (let [max-axes (nd/max array (int-array [1]))
            expected (nd/create ndm (float-array [4 8]))]
        (is (= max-axes expected)))
      (let [max-keep (nd/max array (int-array [0]) true)
            expected (nd/create ndm (float-array [6 8]) (nd/shape 1 2))]
        (is (= max-keep expected))))
    (let [array (nd/create ndm 5.)]
      (is (= (nd/get-element (nd/max array)) 5.)))
    ;; MXNet engine call failed: MXNetError: Check failed
    ;; is_all_reducded_axes_not_zero: zero-size array to reduction operation maximum
    ;; which has no identity Stack trace: File
    ;; "src/operator/numpy/np_broadcast_reduce_op.h", line 231
    #_(let [array (nd/create ndm (nd/shape 1 0))]
        (nd/max array))))

(deftest min-test
  (with-open [ndm (nd/new-base-manager)]
    (let [array (nd/create ndm (float-array [2 1 5 0]))]
      (is (= (nd/get-element (nd/min array)) 0.)))
    (let [array (nd/create ndm (float-array [2 4 6 8]) (nd/shape 2 2))]
      (is (= (nd/get-element (nd/min array)) 2.))
      (let [min-axes (nd/min array (int-array [1]))
            expected (nd/create ndm (float-array [2 6]))]
        (is (= min-axes expected)))
      (let [min-keep (nd/min array (int-array [0]) true)
            expected (nd/create ndm (float-array [2 4]) (nd/shape 1 2))]
        (is (= min-keep expected))))
    (let [array (nd/create ndm 5.)]
      (is (= (nd/get-element (nd/min array)) 5.)))
    ;; MXNet engine call failed: MXNetError: Check failed
    ;; is_all_reducded_axes_not_zero: zero-size array to reduction operation maximum
    ;; which has no identity Stack trace: File
    ;; "src/operator/numpy/np_broadcast_reduce_op.h", line 231
    #_(let [array (nd/create ndm (nd/shape 1 0))]
        (nd/min array))))

(deftest sum-test
  (with-open [ndm (nd/new-base-manager)]
    (let [array (nd/create ndm (float-array [1 2 3 5]))]
      (is (= (nd/get-element (nd/sum array)) 11.)))
    (let [array (nd/create ndm (float-array [2 4 6 8]) (nd/shape 2 2))]
      (is (= (nd/get-element (nd/sum array)) 20.))
      (let [sum-axes (nd/sum array (int-array [1]))
            expected (nd/create ndm (float-array [6 14]))]
        (is (= sum-axes expected)))
      (let [sum-keep (nd/sum array (int-array [0]) true)
            expected (nd/create ndm (float-array [8 12]) (nd/shape 1 2))]
        (is (= sum-keep expected))))
    (let [array (nd/create ndm 5.)]
      (is (= (nd/get-element (nd/sum array)) 5.)))
    ;; zero dim
    (let [array (nd/create ndm (nd/shape 1 0 0))]
      (is (= (nd/get-element (nd/sum array)) 0.)))))

(deftest prod-test
  (with-open [ndm (nd/new-base-manager)]
    (let [array (nd/create ndm (float-array [1 2 3 4]))]
      (is (= (nd/get-element (nd/prod array)) 24.)))
    (let [array (nd/create ndm (float-array [2 4 6 8]) (nd/shape 2 2))]
      (is (= (nd/get-element (nd/prod array)) 384.))
      (let [prod-axes (nd/prod array (int-array [1]))
            expected (nd/create ndm (float-array [8 48]))]
        (is (= prod-axes expected)))
      (let [prod-keep (nd/prod array (int-array [0]) true)
            expected (nd/create ndm (float-array [12 32]) (nd/shape 1 2))]
        (is (= prod-keep expected))))
    (let [array (nd/create ndm 5.)]
      (is (= (nd/get-element (nd/prod array)) 5.)))
    ;; zero dim
    (let [array (nd/create ndm (nd/shape 1 0 0))]
      (is (= (nd/get-element (nd/prod array)) 0.)))))

(deftest mean-test
  (with-open [ndm (nd/new-base-manager)]
    (let [array (nd/create ndm (float-array [1 2 3 4]))]
      (is (= (nd/get-element (nd/mean array)) 2.5)))
    (let [array (nd/create ndm (float-array [2 4 6 8]) (nd/shape 2 2))]
      (is (= (nd/get-element (nd/mean array)) 5.))
      (let [mean-axes (nd/mean array (int-array [1]))
            expected (nd/create ndm (float-array [3 7]))]
        (is (= mean-axes expected)))
      (let [mean-keep (nd/mean array (int-array [0]) true)
            expected (nd/create ndm (float-array [4 6]) (nd/shape 1 2))]
        (is (= mean-keep expected))))
    (let [array (nd/create ndm 5.)]
      (is (= (nd/get-element (nd/mean array)) 5.)))
    ;; zero dim
    (let [array (nd/create ndm (nd/shape 1 0 0))]
      (is (= (nd/get-element (nd/mean array)) 0.)))))

(deftest trace-test
  (with-open [ndm (nd/new-base-manager)]
    (let [original (-> (nd/arange ndm 8.) (nd/reshape (nd/shape 2 2 2)) (nd/trace))
          expected (nd/create ndm (float-array [6 8]))]
      (is (= original expected)))
    (let [original (-> (nd/arange ndm 24.) (nd/reshape (nd/shape 2 2 2 3)) (nd/trace))
          expected (nd/create ndm (float-array [6 8]))]
      (is (= (nd/get-shape original) (nd/shape 2 3))))))
