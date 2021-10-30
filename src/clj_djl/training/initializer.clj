(ns clj-djl.training.initializer
  (:import [ai.djl.training.initializer Initializer
            XavierInitializer XavierInitializer$RandomType XavierInitializer$FactorType
            NormalInitializer]))


(defn ones
  "return ONES intializer"
  []
  Initializer/ONES)

(defn zeros
  "return ZEROS initializer"
  []
  Initializer/ZEROS)

(defn xavier
  ([]
   (XavierInitializer.))
  ([^clojure.lang.Keyword random-type ^clojure.lang.Keyword factor-type magnitude]
   (XavierInitializer. (XavierInitializer$RandomType/valueOf (.toUpperCase (name random-type)))
                       (XavierInitializer$FactorType/valueOf (.toUpperCase (name factor-type)))
                       magnitude)))

(def new-xavier xavier)

(defn normal-initializer
  ([]
   (NormalInitializer.))
  ([sigma]
   (NormalInitializer. sigma)))

(def new-normal-initializer normal-initializer)
