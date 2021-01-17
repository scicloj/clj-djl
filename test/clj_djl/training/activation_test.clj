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
  (with-open [model (m/model {:name "model"
                              :block (nn/softplus-block)})
              trainer (t/trainer model config)
              manager (t/get-manager trainer)]
    (let [data (nd/create manager [0. 0. 2.])
          expected (nd/create manager [0.6931 0.6931 2.1269])]
      (is (nd/all-close (nn/softplus data) expected 0.0001 0.0001 true)))))

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
    (with-open [model (m/model  {:name "leaky relu"
                                 :block (nn/leaky-relu-block alpha)})
                trainer (t/trainer model
                                   (t/config {:loss (loss/l2)
                                              :initializer (init/ones)}))
                manager (t/get-manager trainer)]
      (let [data (nd/create manager (float-array [-10 0 2]))
            expect (nd/create manager (float-array [-5 0 2]))]
        (is (= expect (nn/leaky-relu data alpha)))
        (is (= expect (first (nn/leaky-relu (nd/ndlist data) alpha))))
        (is (= expect (first (t/forward trainer (nd/ndlist data)))))))))

(deftest elu-test
  (let [alpha 1.0]
    (with-open [model (m/model  {:name "elu"
                                 :block (nn/elu-block alpha)})
                trainer (t/trainer model
                                   (t/config {:loss (loss/l2)
                                              :initializer (init/ones)}))
                manager (t/get-manager trainer)]
      (let [data (nd/create manager (float-array [-2 0 2]))
            expect (nd/create manager (float-array [-0.8647 0 2]))]
        (is (nd/all-close expect (nn/elu data alpha)
                          0.001 0.001 true))
        (is (nd/all-close expect (first (nn/elu (nd/ndlist data) alpha))
                          0.001 0.001 true))
        (is (nd/all-close expect (first (t/forward trainer (nd/ndlist data)))
                          0.001 0.001 true)))
      (t/close trainer)))

  (let [alpha 0.5]
    (with-open [model (m/model  {:name "elu"
                                 :block (nn/elu-block alpha)})
                trainer (t/trainer model
                                   (t/config {:loss (loss/l2)
                                              :initializer (init/ones)}))
                manager (t/get-manager trainer)]
      (let [data (nd/create manager (float-array [-2 0 2]))
            expect (nd/create manager (float-array [-0.4323 0 2]))]
        (is (nd/all-close expect (nn/elu data alpha)
                          0.001 0.001 true))
        (is (nd/all-close expect (first (nn/elu (nd/ndlist data) alpha))
                          0.001 0.001 true))
        (is (nd/all-close expect (first (t/forward trainer (nd/ndlist data)))
                          0.001 0.001 true))
        (t/close trainer)))))

(deftest selu-test
  (with-open [model (m/model  {:name "selu"
                               :block (nn/selu-block)})
              trainer (t/trainer model
                                 (t/config {:loss (loss/l2)
                                            :initializer (init/ones)}))
              manager (t/get-manager trainer)]
    (let [data   (nd/create manager (float-array [-2 0 2]))
          expect (nd/create manager (float-array [-1.5202 0 2.1014]))]
      (is (nd/all-close expect (nn/selu data)
                        0.001 0.001 true))
      (is (nd/all-close expect (first (nn/selu (nd/ndlist data)))
                        0.001 0.001 true))
      (is (nd/all-close expect (first (t/forward trainer (nd/ndlist data)))
                        0.001 0.001 true)))))

(deftest gelu-test
  (with-open [model (m/model  {:name "gelu"
                               :block (nn/gelu-block)})
              trainer (t/trainer model
                                 (t/config {:loss (loss/l2)
                                            :initializer (init/ones)}))
              manager (t/get-manager trainer)]
    (let [data   (nd/create manager (float-array [-2 0 2]))
          expect (nd/create manager (float-array [-0.0454 0 1.9546]))]
      (is (nd/all-close expect (nn/gelu data)
                        0.001 0.001 true))
      (is (nd/all-close expect (first (nn/gelu (nd/ndlist data)))
                        0.001 0.001 true))
      (is (nd/all-close expect (first (t/forward trainer (nd/ndlist data)))
                        0.001 0.001 true)))))

(deftest swish-test
  (let [beta 1.0]
    (with-open [model (m/model  {:name "swish"
                                 :block (nn/swish-block beta)})
                trainer (t/trainer model
                                   (t/config {:loss (loss/l2)
                                              :initializer (init/ones)}))
                manager (t/get-manager trainer)]
      (let [data   (nd/create manager (float-array [1 5 0.3 0.08]))
            expect (nd/create manager (float-array [0.7311 4.9665 0.1723 0.0416]))]
        (is (nd/all-close expect (nn/swish data beta)
                          0.001 0.001 true))
        (is (nd/all-close expect (first (nn/swish (nd/ndlist data) beta))
                          0.001 0.001 true))
        (is (nd/all-close expect (first (t/forward trainer (nd/ndlist data)))
                          0.001 0.001 true))))))

(deftest mish-test
  (with-open [model (m/model  {:name "mish"
                               :block (nn/mish-block)})
              trainer (t/trainer model
                                 (t/config {:loss (loss/l2)
                                            :initializer (init/ones)}))
              manager (t/get-manager trainer)]
    (let [data   (nd/create manager (float-array [1, 5, 0.3, 0.08]))
          expect (nd/create manager (float-array [0.9558, 5, 0.253, 0.0628]))]
      (is (nd/all-close expect (nn/mish data)
                        0.001 0.001 true))
      (is (nd/all-close expect (first (nn/mish (nd/ndlist data)))
                        0.001 0.001 true))
      (is (nd/all-close expect (first (t/forward trainer (nd/ndlist data)))
                        0.001 0.001 true)))))

(deftest prelu-test
  (with-open [model (m/model  {:name "mish"
                               :block (nn/prelu-block)})
              trainer (t/trainer model
                                 (t/config {:loss (loss/l2)
                                            :initializer (init/ones)}))
              manager (t/get-manager trainer)]
    (t/initialize trainer (nd/shape 3))
    (let [data   (nd/create manager (float-array [-1, 0, 2]))
          expect (nd/create manager (float-array [-1, 0, 2]))]
      (is (nd/all-close expect (first (t/forward trainer (nd/ndlist data)))
                        0.001 0.001 true)))))
