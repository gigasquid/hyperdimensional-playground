(ns hyperdimensional-playground.context-vectors
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.linear :as ml]
            [clojure.java.io :as io]
            [clojure.string :as string])
  (:import (java.util Properties)
           (edu.stanford.nlp.pipeline StanfordCoreNLP Annotation)
           (edu.stanford.nlp.ling CoreAnnotations$SentencesAnnotation CoreAnnotations$TokensAnnotation
                                  CoreAnnotations$TextAnnotation CoreAnnotations$PartOfSpeechAnnotation
                                  CoreAnnotations$NamedEntityTagAnnotation)))



(defn cosine-sim [v1 v2]
  (let [norm1 (ml/norm v1)
        norm2 (ml/norm v2)]
   (when (and (pos? norm1) ( pos? norm2))
     (/ (m/dot v1 v2)
        (* (ml/norm v1) (ml/norm v2))))))

(defn mean-add [& hvs]
  (m/emap #(Math/round %)
   (m/div (apply m/add hvs) (count hvs))))

(defn xor-mul [v1 v2]
  (->> (m/add v1 v2)
      (m/emap #(mod % 2))))

(defn hamming-dist [v1 v2]
  (m/esum (xor-mul v1 v2)))



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
    (set (map :token nouns))))

(defn gather-nouns-from-book [book-str]
  (let [book-text (slurp book-str)
        docs (string/split book-text #"\n")]
    (reduce (fn [nouns doc]
              (clojure.set/union nouns (filter-nouns doc)))
            #{}
            docs)))

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
(count fairy-tales-nouns) ;=> 11685


;;;; construct a matrix of 11685 x 10,000 and use random columns of 20
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


(m/set-current-implementation :vectorz)
(def sz 10000)
(def noun-idx (zipmap fairy-tales-nouns (range)))
;; rows are the distinct words cols are the 12 word partition windows
(def freq-matrix (m/new-sparse-array [(count fairy-tales-nouns) sz]))


(defn update-doc! [doc]
  (let [known-nouns (clojure.set/intersection fairy-tales-nouns (filter-nouns doc))]
    (doall (repeatedly 20 #(doseq [noun known-nouns]
                       (m/mset! freq-matrix (get noun-idx noun) (rand-int sz) 1))))))

(defn process-book [book-str]
  (let [book-text (slurp book-str)
        docs (string/split book-text #"\n")
        doc-count (count docs)]
    (println "processing:" book-str "(with" doc-count "docs)")
    (doall (map-indexed (fn [idx doc]
                          (println "doc:" idx)
                          (update-doc! doc))
                        docs))
    (println "DONE with " book-str)))


(defn sim-report []
  (for [word1 (take 1000 fairy-tales-nouns)
        word2 (take 1000 fairy-tales-nouns)]
    (when (not= word1 word2)
      {:word1 word1
       :word2 word2
       :cosine (cosine-sim (m/slice freq-matrix (get noun-idx word1))
                           (m/slice freq-matrix (get noun-idx word2)))})))

(reduce + (for [row freq-matrix]
   (m/esum row)))


(comment

  (doseq [book book-list]
    (process-book book))



 ;(process-book "resources/grimm_fairy_tales.txt")

 (def x (remove nil? (sim-report)))
 (second x)
 (take 100 x)
 (sort-by :cosine (filter #( :cosine % ) x))
 (def results (sort-by :cosine (filter #(and (:cosine %) (< 0.01 (:cosine %))) x)))

 (take 100 (reverse results))
)
