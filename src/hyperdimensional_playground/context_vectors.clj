(ns hyperdimensional-playground.context-vectors
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.linear :as ml]
            [clojure.java.io :as io]
            [clojure.string :as string]))

(m/set-current-implementation :vectorz)

(defn cosine-sim [v1 v2]
  (/ (m/dot v1 v2)
     (* (ml/norm v1) (ml/norm v2))))

(defn mean-add [& hvs]
  (m/emap #(Math/round %)
   (m/div (apply m/add hvs) (count hvs))))

(defn xor-mul [v1 v2]
  (->> (m/add v1 v2)
      (m/emap #(mod % 2))))

(defn hamming-dist [v1 v2]
  (m/esum (xor-mul v1 v2)))

;; context vector for word is computed from surround text
;; context window (a dozen words)
;; multiset of words (bag of words) in all the context windows for a
;; given word  (can also do multiset of documents for word)

; maxtrix of frequencies - each row has a word - column is row of
; vocabulary (col per row ) or row of document - transformed Latent
; Sementic Analysis (LSA)

;; use Random indexing
;; collect 100,000 x10,000 matrix
;; each word in the vocabulary has its own raw
;; each document is assigned small numer of columns in out of 10,000
;; (say 20 random) - the document activates those columns.  Each word
;; adds a 1 in all the 20 cols that the document activates


(def grimm (slurp "resources/grimm_fairy_tales.txt"))
;; break up into our documents
(def words (string/split grimm #" |\n|-|;|:|\!|\.|\?|\(|\)|\,"))
(def dword-count (count (distinct words)))              ;=> 10168
(def docs (partition-all 12 words))
(def doc-count (count docs))  ;=> 8531

;; calculate it the old fashion way first since it is not "too" big
(def dwords (distinct words))
(def dword-idx (zipmap dwords (range)))

;; rows are the distinct words cols are the 12 word partition windows
(def freq-matrix (m/new-sparse-array [dword-count doc-count]))

(defn update-doc-word-freqs [doc column-idx]
  (println "processing " column-idx)
  (doseq [[word freq] (frequencies doc)]
    (let [i (get dword-idx word)
          j column-idx]
     (m/mset! freq-matrix i j 1))))

(defn update-word-freqs [docs]
  (map-indexed (fn [idx itm] (update-doc-word-freqs itm idx)) docs))

(update-word-freqs docs)
;;;;;

;; (m/esum (m/slice freq-matrix 1))

;; (take 40 dword-idx)
;; (def merry (m/slice freq-matrix 6853))
;; (def mayor (m/slice freq-matrix 6088))
;; (def weary (m/slice freq-matrix 4430))
;; (def happy (m/slice freq-matrix 4377))

;(m/esum happy)
;(get (frequencies words) "merry") ;=> 19
;(filter #(get % "merry") (map frequencies  docs))
;(m/esum (m/slice freq-matrix (get dword-idx "merry"))) ;=> 19




(defn sim-report []
  (for [word1 (take 1000 dwords)
        word2 (take 1000 dwords)]
    (when (not= word1 word2)
      {(str word1 "-" word2)
       (cosine-sim (m/slice freq-matrix (get dword-idx word1))
                   (m/slice freq-matrix (get dword-idx word2)))})))

(def x (remove nil? (sim-report)))
(ffirst x)
(reduce merge (filter #(< 0.5 (last (first %))) x))
