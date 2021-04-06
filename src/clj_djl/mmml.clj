(ns clj-djl.mmml
  (:require [clj-djl.model :as m]
            [clj-djl.ndarray :as nd]
            [clj-djl.nn :as nn]
            [clj-djl.training :as t]
            [clj-djl.training.dataset :as ds]
            [scicloj.metamorph.ml :as ml]
            [tech.v3.dataset :as dataset])
  (:import java.io.File))

(defn ->ndarray
  "Convert dataframe to NDArray"
  [ndm dataframe]
  (nd/t (nd/create ndm (map vec (dataset/columns dataframe)))))

(defn- train
  [feature-ds label-ds options]
  (let [ndm (nd/new-base-manager)
        train-nd (->ndarray ndm feature-ds)
        label-nd (->ndarray ndm label-ds)
        train-dataset
        (ds/array-dataset {:data (nd/to-type train-nd :float32 false)
                           :labels (nd/to-type label-nd :float32 false)
                           :batchsize (options :batchsize)
                           :shuffle false})
        temp-file (File/createTempFile "model" ".mm")
        temp-dir (.getParent temp-file )
        temp-file-name (.getName temp-file)
        model-spec (:model-spec options)]

    (with-open [model (m/model {:name (:name model-spec) :block ((:block-fn model-spec))}
                               )
                trainer (t/trainer model (:model-cfg options))]
      (t/initialize trainer (:initial-shape options))
      (t/set-metrics trainer (t/metrics))
      (t/fit trainer (:nepoch options) train-dataset)
      (m/save model temp-dir temp-file-name)
      (merge options {
        :train-result (t/get-result trainer)
        :model-file (.getAbsolutePath  temp-file)
        :model-spec (:model-spec options)}
       ))))

(defn- predict
  [feature-ds thawed-model {:keys [target-columns target-categorical-maps options model-data] :as model}]
  (let [ndm (nd/new-base-manager)
        translator (ai.djl.translate.NoopTranslator. nil)
        model-spec (:model-spec options)
        prediction
        (with-open [model (m/model {:name (:name model-spec) :block ((:block-fn model-spec))} )]
          (let [loaded-model (m/load model
                                     (.getParent (clojure.java.io/file (model-data :model-file)))
                                     (.getName (clojure.java.io/file (model-data :model-file))))
                predictor (m/new-predictor model translator)
                test-nd (->ndarray ndm feature-ds)]
            (->
             (.predict predictor (nd/ndlist (nd/to-type test-nd :float32 false)))
             (nd/get 0)
             nd/to-vec)))]
    (dataset/->dataset
     {(first target-columns)
      prediction})))

(ml/define-model! :clj-djl/djl
  train
  predict
  {:documentation {:user-guide "https://github.com/awslabs/djl/blob/master/docs/README.md#documentation"
                   :javadoc "https://javadoc.io/doc/ai.djl/api/latest/index.html"
                   }
   :options [{:name "batchsize" :type :int16 :default nil}
                             {:name "model-spec" :type :model-spec :default nil}
                             {:name "name" :type :string :default nil}
                             {:name "initial-shape" :type :shape :default nil}
                             {:name "nepoch" :type :int16 :default nil}
                             ]
   }
  )
