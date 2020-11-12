(ns clj-djl.ndarray.ndarray-numeric-test
  (:require [clojure.test :refer :all]
            [clj-djl.ndarray :as nd]))
(def ndm (nd/new-base-manager))
(deftest negation
  (with-open [ndm (nd/new-base-manager)]
    (let [data [6 9 -12 -11 0]
          array (nd/create ndm data)
          data (map - data)
          expected (nd/create ndm data)]
      (is (= (nd/- array) expected))
      (nd/-! array)
      (is (= array expected)))
    ;;multi-dim
    (let [data [-2.2 2.2 3 -0.2 2.76 0.002]
          array (nd/create ndm data [2 3])
          data (map - data)
          expected (nd/create ndm data [2 3])]
      (is (= (nd/- array) expected))
      (nd/-! array)
      (is (= array expected)))
    ;;scalar
    (let [array (nd/create ndm 3.)
          expected (nd/create ndm -3.)]
      (is (= (nd/- array) expected))
      (nd/-! array)
      (is (= array expected)))
    ;;zero-dim
    (let [array (nd/create ndm nil [2 0 1])
          expected (nd/create ndm nil [2 0 1])]
      (is (= (nd/- array) expected))
      (nd/-! array)
      (is (= array expected)))))
