(ns clj-djl.ndarray-test
  (:require [clojure.test :refer :all]
            [clj-djl.ndarray :as nd]))

(deftest size-test
  (testing "ndarray/size."
    (is (= 0 (nd/size (nd/arange 0 0))))
    (is (= 100 (nd/size (nd/reshape (nd/arange 0 100)
                                    [10 10]))))))
