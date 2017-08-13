#!/bin/bash
cats='cat'
dogs='dog'
for filename in * ; 
do
	case "$filename" in 
  *cat*)
    mv $filename Cats/
    ;;
esac
done