(ns clj-djl.training.initializer
  (:import [ai.djl.training.initializer Initializer
            XavierInitializer XavierInitializer$RandomType XavierInitializer$FactorType]))

(def ones Initializer/ONES)

(def zeros Initializer/ZEROS)

(defn new-xavier
  ([]
   (XavierInitializer.))
  ([^clojure.lang.Keyword random-type ^clojure.lang.Keyword factor-type magnitude]
   (XavierInitializer. (XavierInitializer$RandomType/valueOf (.toUpperCase (name random-type)))
                       (XavierInitializer$FactorType/valueOf (.toUpperCase (name factor-type)))
                       magnitude)))
