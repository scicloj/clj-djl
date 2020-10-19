(ns clj-djl.engine
  (:import [ai.djl.engine Engine]))

(defn get-all-engines []
  (Engine/getAllEngines))

(defn get-engine-name [engine]
  (.getEngineName engine))

(defn get-engine [engine-name]
  (Engine/getEngine engine-name))

(defn get-instance []
  (Engine/getInstance))

(defn get-version [engine]
  (.getVersion engine))

(defn has-capability [engine capability]
  (.hasCapability engine capability))

(defn has-engine [engine-name]
  (Engine/hasEngine engine-name))

(defn new-gradient-collector [engine]
  (.newGradientCollector engine))
