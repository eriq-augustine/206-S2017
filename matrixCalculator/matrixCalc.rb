# There is no error checking around here, so input everything correctly.

DEFAULT_TABLEAU = false

PADDING = 7
PRECISION = 2
ROUNDING_PRECISION = 5
EPSILON = 0.00001

def addToBasis(matrix, oneRow, basisCol)
   oneRow = oneRow.to_i()
   basisCol = basisCol.to_i()

   # First, one the target cell.
   oneCell(matrix, oneRow, basisCol)

   # Then zero each other cell in the col with row addition.
   matrix.each_index{|row|
      if (row == oneRow)
         next
      end

      addRow(matrix, oneRow, row, -1.0 * matrix[row][basisCol])
   }

   normalize(matrix)
end

def oneCell(matrix, row, col)
   row = row.to_i()
   col = col.to_i()

   scaleFactor = 1.0 / matrix[row][col]

   scaleRow(matrix, row, scaleFactor)
end

def addRow(matrix, sourceRow, destRow, scaleFactor = 1.0)
   sourceRow = sourceRow.to_i()
   destRow = destRow.to_i()
   scaleFactor = scaleFactor.to_f()

   matrix[sourceRow].each_index{|col|
      matrix[destRow][col] += matrix[sourceRow][col] * scaleFactor
   }

   normalize(matrix)
end

def scaleRow(matrix, row, scaleFactor)
   row = row.to_i()
   scaleFactor = scaleFactor.to_f()

   matrix[row].each_index{|col|
      matrix[row][col] *= scaleFactor
   }

   normalize(matrix)
end

def floatEquals(a, b)
   return (a - b).abs() < EPSILON
end

# Try to find ints,
def normalize(matrix)
   matrix.each_index{|row|
      matrix[row].each_index{|col|
         val = matrix[row][col]
         if (val.is_a?(Integer))
            next
         end

         if (floatEquals(val.round(ROUNDING_PRECISION).to_i(), val))
            matrix[row][col] = val.to_i()
         end
      }
   }
end

def formatNumber(number)
   if (number.is_a?(Integer))
      return "%#{PADDING}d" % number
   end

   return "%#{PADDING}.#{PRECISION}f" % number
end

def printMatrix(matrix, tableau = false)
   if (!tableau)
      puts matrix.map{|row| row.map{|value| formatNumber(value)}.join(" ")}.join("\n")
      return
   end

   lines = []
   matrix.each_index{|row|
      line = []
      matrix[row].each_index{|col|
         line << formatNumber(matrix[row][col])

         if (col == matrix[row].size() - 2)
            line << '|'
         end
      }
      lines << line.join(' ')

      if (row == 0)
         lines << '-' * lines[0].size()
      end
   }
   puts lines.join("\n")
end

# HACK(eriq): Just being lazy.
def parseNumber(text)
   if (text.to_i() == text.to_f())
      return text.to_i()
   end

   return text.to_f()
end

def readMatrix(history)
   print("Num Rows: ")
   rows = gets().strip().to_i()
   history << rows

   print("Num Cols: ")
   cols = gets().strip().to_i()
   history << cols

   matrix = Array.new(rows){ Array.new(cols) }
   for row in 0...rows
      vals = gets().strip().split(' ').map{|val| parseNumber(val.strip())}
      if (vals.size() != cols)
         puts "Incorrect number of cols. Found: #{vals.size()}, required: #{cols}"
         exit 2
      end

      history << vals.join(' ')
      
      for col in 0...cols
         matrix[row][col] = vals[col]
      end
   end

   normalize(matrix)
   return matrix
end

def loadArgs(args)
   if (args.size() > 1 || args.map{|arg| arg.strip().downcase().sub(/^-+/, '')}.include?('help'))
      puts "USAGE: ruby #{$0} [--tableau]"
      puts "Commands:"
      puts "   q[uit] - You're done."
      puts "   s[cale] <row index> <scale factor> - Scale a row."
      puts "   a[dd] <source row> <dest row> [scale factor] - Add a row to another (with possible scaling)."
      puts "   o[ne] <row> <col> - Scale a row so that it has a 1 in a specific cell."
      puts "   b[asis] <row> <col> - Add this column to this basis, using the row as the one."
   end

   tableau = DEFAULT_TABLEAU

   if (args.size() > 0)
      arg = args.shift()
      if (arg == '--tableau')
         tableau = true
      else
         puts "Unknown arg: '#{arg}'"
         exit 1
      end
   end

   return tableau
end

def main(tableau)
   history = []
   matrix = readMatrix(history)

   while (true)
      puts ""
      printMatrix(matrix, tableau)

      args = gets().strip().split(' ').map{|arg| arg.strip()}

      if (args.size() == 0)
         next
      end
      history << args.join(' ')

      command = args.shift().downcase()

      case command
      when 'q', 'quit'
         break
      when 's', 'scale'
         scaleRow(matrix, *args)
      when 'a', 'add'
         addRow(matrix, *args)
      when 'o', 'one'
         oneCell(matrix, *args)
      when 'b', 'basis'
         addToBasis(matrix, *args)
      else
         puts "Unknown command: '#{command}'"
      end
   end

   puts 'Command History:'
   puts history.join("\n")
end

if ($0 == __FILE__)
   main(*loadArgs(ARGV))
end
