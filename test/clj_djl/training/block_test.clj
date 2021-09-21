(ns clj-djl.training.block-test
  (:require [clojure.test :refer [deftest is]]
            [clj-djl.ndarray :as nd]
            [clj-djl.training :as t]
            [clj-djl.model :as m]
            [clj-djl.nn :as nn]
            [clj-djl.training.loss :as loss]
            [clj-djl.training.initializer :as init]
            [clj-djl.nn.parameter :as param])
  (:import (ai.djl.training.initializer Initializer)))

(deftest flatten-block
  (let [config (-> (loss/l2-loss) (t/new-default-training-config) (t/opt-initializer Initializer/ONES param/weight))]
    (with-open [model (-> (m/new-instance "model")
                          (m/set-block (nn/batch-flatten-block)))
                trainer (m/new-trainer model config)]
      (let [manager (t/get-manager trainer)
            param-store (t/parameter-store manager false)
            data (nd/random-uniform manager 0 255 [10 28 28])
            expected (nd/reshape data [10 (* 28 28)])
            result (-> model m/get-block (nn/forward param-store (nd/ndlist data) true) nd/head)]
        (is (= result expected))))))

(deftest linear-block
  (let [cfg (t/config {:loss (loss/l2) :initializer (init/ones) :parameter param/weight})
        out-size 3
        input-shape (nd/shape 2 2)]
    (with-open [model (m/model {:name "model"
                                :block (nn/linear {:units out-size})})
                trainer (t/trainer model cfg)]
      (t/initialize trainer input-shape)
      (let [ndm (t/get-manager trainer)
            data (nd/create ndm (float-array [1 2 3 4]) input-shape)
            result (t/forward trainer (nd/ndlist data))
            expected (-> data
                         (nd/dot (nd/transpose (nd/ones ndm (nd/shape out-size 2))))
                         (nd/+ (nd/zeros ndm (nd/shape 2 out-size))))]
        (is (= (first result) expected))))

    (with-open [model (m/model {:name "model"
                                :block (nn/linear {:units out-size :bias false})})
                trainer (t/trainer model cfg)]
      (t/initialize trainer input-shape)
      (let [ndm (t/get-manager trainer)
            data (nd/create ndm (float-array [1 2 3 4]) input-shape)
            result (t/forward trainer (nd/ndlist data))
            expected (-> data
                         (nd/dot (nd/transpose (nd/ones ndm (nd/shape out-size 2))))
                         (nd/+ (nd/zeros ndm (nd/shape 2 out-size))))]
        (is (= (first result) expected))))

    (let [out-size 10
          input-shape (nd/shape 10 20 12)]
      (with-open [model (m/model {:name "model"
                                  :block (nn/linear {:units out-size})})
                  trainer (t/trainer model cfg)]
        (t/initialize trainer input-shape)
        (let [ndm (t/get-manager trainer)
              data (nd/ones ndm input-shape)
              result (t/forward trainer (nd/ndlist data))]
          (is (= (nd/shape (first result)) (nd/shape 10 20 10))))))))



(comment
  (def ndm (nd/new-base-manager))
  (-> (nd/create ndm (float-array [1 2 3 4]) [2 2])
      (nd/dot (nd/transpose (nd/ones ndm (nd/shape 3 2))))
      (nd/+ (nd/zeros ndm [2 3]))))
