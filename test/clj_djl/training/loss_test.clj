(ns clj-djl.training.loss-test
  (:require [clojure.test :refer :all]
            [clj-djl.ndarray :as nd]
            [clj-djl.training.loss :as loss]))

(deftest l1-loss
  (with-open [ndm (nd/new-base-manager)]
    (let [pred  (nd/create ndm (float-array [1 2 3 4 5]))
          label (nd/ones ndm [5])]
      ;; L1 = \sum |pred_i - label_i| = 0+1+2+3+4 = 10; 10/5 = 2
      (is (> 1e-4 (Math/abs (- 2.0
                               (-> (loss/l1)
                                   (loss/evaluate label pred)
                                   (nd/get-element)))))))))

(deftest l2-loss
  (with-open [ndm (nd/new-base-manager)]
    (let [pred  (nd/create ndm (float-array [1 2 3 4 5]))
          label (nd/ones ndm [5])]
      ;; L2 = 0.5 * \sum |label_i - pred_i|^2 = 1/2 * (1+4+9+16) = 15; 15/5 = 3
      (is (> 1e-4 (Math/abs (- 3.
                               (-> (loss/l2)
                                   (loss/evaluate label pred)
                                   (nd/get-element)))))))))

(deftest hinge-loss
  (with-open [ndm (nd/new-base-manager)]
    (let [pred  (nd/create ndm (float-array [1 2 3 4 5]))
         label (nd/create ndm (int-array [1 1 -1 -1 1]))]
        ;; L = \sum {max(0, 0), max(0, -1), max(0 4), max(0, 5), max(0, -4)} = 9; 9/5 = 1.8
        (is (> 1e-4 (Math/abs (- 1.8
                                 (-> (loss/hinge)
                                     (loss/evaluate label pred)
                                     (nd/get-element)))))))))

(deftest softmax-cross-entropy-loss
  (with-open [ndm (nd/new-base-manager)]
    (let [pred  (nd/create ndm (float-array [1 2 3 4 5]))
          label (nd/ones ndm [1])]
      (is (> 1e-4 (Math/abs (- 3.45191431
                               (-> (loss/sotfmax-cross-entropy)
                                   (loss/evaluate label pred)
                                   (nd/get-element)))))))))

(deftest sigmoid-binary-cross-entropy-loss
  (with-open [ndm (nd/new-base-manager)]
    (let [pred  (nd/create ndm (float-array [1 2 3 4 5]))
          label (nd/ones ndm [5])]
      (is (> 1e-4 (Math/abs (- 0.10272846
                               (-> (loss/sigmoid-binary-cross-entropy)
                                   (loss/evaluate label pred)
                                   (nd/get-element)))))))))

(deftest masked-softmax-cross-entropy-loss
  (with-open [ndm (nd/new-base-manager)]
    (let [pred  (nd/ones ndm [3 4 10])
          label (nd/ones ndm [3 4])
          valid-lengths (nd/create ndm (int-array [4 2 0]))]
      (is (-> (loss/masked-softmax-cross-entropy)
              (loss/evaluate [label valid-lengths] pred)
              (nd/- (nd/create ndm (float-array [2.3025851 1.1512926 0]) (nd/shape [3 1])))
              (nd/sum)
              (nd/get-element)
              (Math/abs)
              (< 1e-4))))))
