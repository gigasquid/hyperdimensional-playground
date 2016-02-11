(ns hyperdimensional-playground.fairytale-nouns
  (:require [clojure.string :as string])
  (:import (java.util Properties)
           (edu.stanford.nlp.pipeline StanfordCoreNLP Annotation)
           (edu.stanford.nlp.ling CoreAnnotations$SentencesAnnotation CoreAnnotations$TokensAnnotation
                                  CoreAnnotations$TextAnnotation CoreAnnotations$PartOfSpeechAnnotation
                                  CoreAnnotations$NamedEntityTagAnnotation)))

;;; get all the nouns in our corpus
(def props (Properties.))
(.setProperty props "annotators", "tokenize, ssplit, pos")

(def pipeline (StanfordCoreNLP. props))

(defn ->text-data [tokens sent-num]
  (mapv (fn [t] {:sent-num sent-num
                :token (.get t CoreAnnotations$TextAnnotation)
                :pos  (.get t CoreAnnotations$PartOfSpeechAnnotation)}) tokens))

(defn process-text [text]
  (let [annotation (Annotation. text)
        _ (.annotate pipeline annotation)
        sentences (.get annotation CoreAnnotations$SentencesAnnotation)
        sentence-tokens (mapv (fn [s] (.get s CoreAnnotations$TokensAnnotation)) sentences)
        text-data (flatten (map-indexed (fn [i t] (->text-data t i)) sentence-tokens))]
    text-data))

(defn filter-nouns [doc]
  (let [nouns (filter #(contains? (hash-set "NN" "NNS" "NNP" "NNPS") (:pos %))
                      (process-text doc))]
    (remove #{"and" "the" "is" "a" "i" "he" "she" "it"}
            (set (map #(clojure.string/lower-case (:token %)) nouns)))))

(defn gather-nouns-from-book [book-str]
  (let [book-text (slurp book-str)
        docs (string/split book-text #"\s|\.|\,|\;|\!|\?")]
    (doall (reduce (fn [nouns doc]
               (clojure.set/union nouns (filter-nouns doc)))
             #{}
             docs))))
(comment
  (def grimm-nouns (gather-nouns-from-book "resources/grimm_fairy_tales.txt"))
  (def anderson-nouns (gather-nouns-from-book "resources/anderson_fairy_tales.txt"))
  (def english-fairy-tale-nouns (gather-nouns-from-book "resources/english_fairy_tales.txt"))
  (def every-child-nouns (gather-nouns-from-book "resources/fairy_tales_every_child_should_know.txt"))
  (def firelight-nouns (gather-nouns-from-book "resources/firelight_fairy_book.txt"))
  (def favorite-nouns (gather-nouns-from-book "resources/favorite_fairy_tales.txt"))
  (def wonder-wings-nouns (gather-nouns-from-book "resources/wonder_wings.txt"))
  (def old-fashioned-nouns (gather-nouns-from-book "resources/old_fashioned.txt"))
  (def fairy-godmother-nouns (gather-nouns-from-book "resources/fairy_godmothers.txt"))
  (def golden-spears-nouns (gather-nouns-from-book "resources/golden_spears.txt"))
  (def red-cap-nouns (gather-nouns-from-book "resources/red_cap_tales.txt"))


  (def fairy-tales-nouns (clojure.set/union grimm-nouns anderson-nouns english-fairy-tale-nouns
                                            every-child-nouns firelight-nouns favorite-nouns
                                            wonder-wings-nouns old-fashioned-nouns fairy-godmother-nouns
                                            golden-spears-nouns red-cap-nouns))

  (take 100 grimm-nouns)
  (count grimm-nouns)
  (count fairy-tales-nouns) ;=> 136606
  (spit "resources/fairy-tales-nouns.edn" fairy-tales-nouns)
  (spit "resources/fairy-tales-nouns-grimm.edn" grimm-nouns)
  (take 100 fairy-tales-nouns)

 (def x (clojure.edn/read-string (slurp "resources/fairy-tales-nouns.edn")))

)
(def grimm-nouns (clojure.edn/read-string (slurp "resources/fairy-tales-nouns-grimm.edn")))
(def fairy-tales-nouns (set grimm-nouns))


;;;; construct a matrix of 10322 x 10,000 and use random columns of 20
;;;; to put frequenies

(def book-list ["resources/grimm_fairy_tales.txt"
                "resources/anderson_fairy_tales.txt"
                "resources/english_fairy_tales.txt"
                "resources/fairy_tales_every_child_should_know.txt"
                "resources/firelight_fairy_book.txt"
                "resources/favorite_fairy_tales.txt"
                "resources/wonder_wings.txt"
                "resources/old_fashioned.txt"
                "resources/fairy_godmothers.txt"
                "resources/golden_spears.txt"
                "resources/red_cap_tales.txt"])
