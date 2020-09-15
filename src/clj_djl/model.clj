(ns clj-djl.model
  (:import [ai.djl Model]))

(defn new-instance [name]
  (Model/newInstance name))

(defn set-block [model block]
  (.setBlock model block)
  model)

(defn set-property [model k v]
  (.setProperty model k v)
  model)

(defn save [model dir name]
  (.save model dir name)
  model)
