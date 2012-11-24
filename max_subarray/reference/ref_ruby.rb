def max_contiguous_submatrix_n3(m)
  rows = m.count
  cols = rows ? m.first.count : 0

  vps = Array.new(rows)
  for i in 0..rows
    vps[i] = Array.new(cols, 0)
  end

  for j in 0...cols
    vps[0][j] = m[0][j]
    for i in 1...rows
      vps[i][j] = vps[i-1][j] + m[i][j]
    end
  end

  max = [m[0][0],0,0,0,0] # this is the result, stores [max,top,left,bottom,right]
  # these arrays are used over Kandane
  sum = Array.new(cols) # obvious sum array used in Kandane
  pos = Array.new(cols) # keeps track of the beginning position for the max subseq ending in j

  for i in 0...rows
    for k in i...rows
      # Kandane over all columns with the i..k rows
      sum.fill(0) # clean both the sum and pos arrays for the upcoming kandane
      pos.fill(0)
      local_max = 0 # we keep track of the position of the max value over each Kandane's execution
      # notice that we do not keep track of the max value, but only its position
      sum[0] = vps[k][0] - (i==0 ? 0 : vps[i-1][0])
      for j in 1...cols
        value = vps[k][j] - (i==0 ? 0 : vps[i-1][j])
        if sum[j-1] > 0
          sum[j] = sum[j-1] + value
          pos[j] = pos[j-1]
        else
          sum[j] = value
          pos[j] = j
        end
        if sum[j] > sum[local_max]
          local_max = j
        end
      end
      # Kandane ends here

      # Here's the key thing
      # If the max value obtained over the past kandane's execution is larger than
      # the current maximum, then update the max array with sum and bounds
      if sum[local_max] > max[0]
        # sum[local_max] is the new max value
        # the corresponding submatrix goes from rows i..k.
        # and from columns pos[local_max]..local_max
        # the array below contains [max_sum,top,left,bottom,right]
        max = [sum[local_max], i, pos[local_max], k, local_max]
      end
    end
  end

  return max # return the array with [max_sum,top,left,bottom,right]
end
