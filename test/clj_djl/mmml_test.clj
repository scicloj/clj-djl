(ns clj-djl.mmml-test
  (:require [clj-djl.mmml]
            [scicloj.metamorph.ml :as ml]
            [tech.v3.dataset  :as ds]
            [tech.v3.dataset.column-filters  :as ds-cf]
            [tech.v3.dataset.modelling  :as ds-mod]
            [clojure.test :refer [deftest is]]
            [clj-djl.nn :as nn]
            [clj-djl.training :as t]
            [clj-djl.training.loss :as loss]
            [clj-djl.training.optimizer :as optimizer]
            [clj-djl.training.tracker :as tracker]
            [clj-djl.training.listener :as listener]
            [clj-djl.ndarray :as nd]
            [tech.v3.datatype.functional :as dfn]
            [clj-djl.nn.parameter :as param]))

(defn count-small [seq]
  (count
   (filter
    #(and (> 5 %)
          (<= -5 %))
    seq)))


(def train-ds
  (ds/->dataset
   "http://d2l-data.s3-accelerate.amazonaws.com/kaggle_house_pred_train.csv"))


(def test-ds
  (->
   (ds/->dataset
    "http://d2l-data.s3-accelerate.amazonaws.com/kaggle_house_pred_test.csv")
   (ds/add-column (ds/new-column  "SalePrice" 0))))


(defn numeric-features [ds]
  (ds-cf/intersection (ds-cf/numeric ds)
                      (ds-cf/feature ds))
  )

(defn update-columns
  "Update a sequence of columns selected by column name seq or column selector function."
  [dataframe col-name-seq-or-fn update-fn]
  (ds/update-columns dataframe
                     (if (fn? col-name-seq-or-fn)
                       (ds/column-names (col-name-seq-or-fn dataframe))
                       col-name-seq-or-fn)
                     update-fn))



(def  learning-rate 0.05)
(defn net [] (nn/sequential {:blocks (nn/linear {:units 1})
                         :initializer (nn/normal-initializer)
                             :parameter param/weight}))
(defn cfg [] (t/training-config {:loss (loss/l2-loss)
                             :optimizer (optimizer/sgd
                                         {:tracker (tracker/fixed learning-rate)})
                             :evaluator (t/accuracy)
                             :listeners (listener/logging)}))
(defn preprocess [ds ds-indices]
    (-> ds
        (ds/drop-columns ["Id"])
        (ds-mod/set-inference-target "SalePrice")
        (ds/replace-missing ds-cf/numeric :value 0)
        (ds/replace-missing ds-cf/categorical :value "None")
        (update-columns numeric-features
                        #(dfn// (dfn/- % (dfn/mean %))
                                (dfn/standard-deviation %)))
        (ds-mod/set-inference-target "SalePrice")
        (ds/categorical->one-hot ds-cf/categorical)
        (ds/select-rows ds-indices)
        ))

(deftest train-predict


  (let [trained-model
        (-> (preprocess (ds/concat train-ds test-ds)
                        (range (ds/row-count train-ds))
                        )
            (ml/train {:model-type :clj-djl/djl
                       :batchsize 64
                       :model-spec {:name "mlp" :block-fn net}
                       :model-cfg (cfg)
                       :initial-shape (nd/shape 1 310)
                       :nepoch 1}))
        prediction
        (-> (preprocess (ds/concat  test-ds train-ds)
                        (range (ds/row-count test-ds))
                        )
            (ml/predict trained-model

                        ))
    _    (def prediction prediction)
        ]
    (is (= 1459
           (count-small
            (get prediction "SalePrice"))))))
