(ns clj-djl.training.activation-test
  (:require [clojure.test :refer :all]
            [clj-djl.ndarray :as nd]
            [clj-djl.model :as m]
            [clj-djl.training :as t]
            [clj-djl.training.loss :as loss]
            [clj-djl.training.initializer :as init]
            [clj-djl.nn :as nn]))

(def config (t/config {:loss (loss/l2) :initializer (init/ones)}))

(deftest relu-test
  (try (let [model (m/new-instance "model")]
         (m/set-block model (nn/relu-block))
         (try (let [trainer (t/new-trainer model config)]
                (t/initialize trainer [(nd/new-shape [3])])
                (let [manager (t/get-manager trainer)
                      data (nd/create manager (float-array [-1 0 2]))
                      expected (nd/create manager (float-array [0 0 2]))
                      result (nd/singleton-or-throw (t/forward trainer [data]))]
                  (is (= expected (nn/relu data)))
                  (is (= expected result))))))))

(deftest sigmoid-test
  (try (let [model (m/new-instance "model")]
         (m/set-block model (nn/sigmoid-block))
         (try (let [trainer (t/trainer model config)]
                (let [manager (t/get-manager trainer)
                      data (nd/create manager (float-array [0]))
                      expected (nd/create manager (float-array [0.5]))
                      result (first (t/forward trainer [data]))]
                  (is (= expected (nn/sigmoid data)))
                  (is (= expected result))))))))

(deftest tanh-test
  (try (let [model (m/new-instance "model")]
         (m/set-block model (nn/tanh-block))
         (let [trainer (t/trainer model config)]
           (let [manager (t/get-manager trainer)
                 data (nd/create manager (float-array [0]))
                 expected (nd/create manager (float-array [0]))
                 result (first (t/forward trainer [data]))]
             (is (= expected (nn/tanh data)))
             (is (= expected result)))))))

(deftest softrelu-test
  (with-open [model (m/new-model {:name "model"
                                  :block (nn/softplus-block)})
              trainer (t/trainer model config)
              manager (t/get-manager trainer)]
    (let [data (nd/create manager [0. 0. 2.])
          expected (nd/create manager [0.6931 0.6931 2.1269])]
      (nd/all-close (nn/softplus data) expected 0.0001 0.0001 true))))

(deftest leaky-relu-test
  (let [alpha 1.0]
    (with-open [model (m/model {:name "leaky relu"
                                :block (nn/leaky-relu-block alpha)})
                trainer (t/trainer model config)
                manager (t/get-manager trainer)]
      (let [data (nd/create manager (float-array [-1 0 2]))
            expect (nd/create manager (float-array [-1 0 2]))]
        (is (= expect (nn/leaky-relu data alpha)))
        (is (= expect (first (nn/leaky-relu (nd/ndlist data) alpha))))
        (is (= expect (first (t/forward trainer (nd/ndlist data))))))))

  (let [alpha 0.1]
    (with-open [model (m/model {:name "leaky relu"
                                :block (nn/leaky-relu-block alpha)})
                trainer (t/trainer model config)
                manager (t/get-manager trainer)]
      (let [data (nd/create manager (float-array [-10 0 2]))
            expect (nd/create manager (float-array [-1 0 2]))]
        (is (= expect (nn/leaky-relu data alpha)))
        (is (= expect (first (nn/leaky-relu (nd/ndlist data) alpha))))
        (is (= expect (first (t/forward trainer (nd/ndlist data))))))))

  (let [alpha 0.5]
    (with-open [trainer (t/trainer (m/model  {:name "leaky relu"
                                              :block (nn/leaky-relu-block alpha)})
                                   (t/config {:loss (loss/l2)
                                              :initializer (init/ones)}))
                manager (t/get-manager trainer)]
      (let [data (nd/create manager (float-array [-10 0 2]))
            expect (nd/create manager (float-array [-5 0 2]))]
        (is (= expect (nn/leaky-relu data alpha)))
        (is (= expect (first (nn/leaky-relu (nd/ndlist data) alpha))))
        (is (= expect (first (t/forward trainer (nd/ndlist data)))))))))
